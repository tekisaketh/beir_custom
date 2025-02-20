from .. import BaseSearch
from .util import cos_sim, dot_score
import logging
import torch
from typing import Dict, List
import heapq
import numpy as np

# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearch(BaseSearch):
    
    def __init__(self, model,corpus_embeddings: np.ndarray=None,corpus_ids: List[int]=None,query_ids: List[int]=None,query_embeddings: np.ndarray=None, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.corpus_embeddings = corpus_embeddings
        self.corpus_ids = corpus_ids
        self.query_ids = query_ids
        self.query_embeddings = query_embeddings
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    print("called this dense function...")
    
    def search(self,
               top_k: int,
               score_function: str, 
               corpus: Dict[str, Dict[str, str]]=None, 
               queries: Dict[str, str]=None, 
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        if(~isinstance(self.query_embeddings,np.ndarray) and ~isinstance(self.query_ids,list)):    
            print("Encoding Queries...")
            query_ids = list(queries.keys())
            self.results = {qid: {} for qid in query_ids}
            queries = [queries[qid] for qid in queries]
            query_embeddings = self.model.encode_queries(
                queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        
        if(~isinstance(self.corpus_ids,list)):  
            print("Sorting Corpus by document length (Longest first)...")
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            corpus = [corpus[cid] for cid in corpus_ids]
            print("Encoding Corpus in batches... Warning: This might take a while!")
            print("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        
        if(~isinstance(self.corpus_embeddings,np.ndarray)):
            itr = range(0, len(corpus), self.corpus_chunk_size)
            result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
            for batch_num, corpus_start_idx in enumerate(itr):
                
                print("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                # Encode chunk of corpus    
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar, 
                    convert_to_tensor = self.convert_to_tensor
                    )

                # Compute similarites using either cosine-similarity or dot product
                cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
                cos_scores[torch.isnan(cos_scores)] = -1

                # Get top-k values
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
                
                for query_itr in range(len(query_embeddings)):
                    query_id = query_ids[query_itr]                  
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                        if corpus_id != query_id:
                            if len(result_heaps[query_id]) < top_k:
                                # Push item on the heap
                                heapq.heappush(result_heaps[query_id], (score, corpus_id))
                            else:
                                # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                                heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

            for qid in result_heaps:
                for score, corpus_id in result_heaps[qid]:
                    self.results[qid][corpus_id] = score
        else:
            result_heaps = {qid: [] for qid in self.query_ids}  # Keep only the top-k docs for each query
            cos_scores = self.score_functions[score_function](self.query_embeddings, self.corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            for query_itr in range(len(self.query_embeddings)):
                    query_id = self.query_ids[query_itr]                  
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[sub_corpus_id]
                        if corpus_id != query_id:
                            if len(result_heaps[query_id]) < top_k:
                                # Push item on the heap
                                heapq.heappush(result_heaps[query_id], (score, corpus_id))
                            else:
                                # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                                heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

            for qid in result_heaps:
                for score, corpus_id in result_heaps[qid]:
                    self.results[qid][corpus_id] = score

        return self.results 
