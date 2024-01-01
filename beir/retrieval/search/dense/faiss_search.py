from .. import BaseSearch
from .util import save_dict_to_tsv, load_tsv_to_dict
from .faiss_index import FaissBinaryIndex, FaissTrainIndex, FaissHNSWIndex, FaissIndex
import logging
import faiss
import numpy as np
import os
from typing import Dict
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

#Parent class for any faiss search
class DenseRetrievalFaissSearch(BaseSearch):
    
    def __init__(self, model, corpus_embeddings: np.ndarray=None,faiss_ids: np.ndarray=None, query_embeddings: np.ndarray=None,query_ids: np.ndarray=None,batch_size: int = 128, corpus_chunk_size: int = 50000, use_gpu: bool = False, **kwargs):
        print("okay")
        self.model = model
        self.batch_size = batch_size
        self.corpus_embeddings = corpus_embeddings
        self.query_embeddings = query_embeddings
        self.faiss_ids = faiss_ids
        self.query_ids = query_ids
        self.corpus_chunk_size = corpus_chunk_size
        self.score_functions = ['cos_sim','dot']
        self.mapping_tsv_keys = ["beir-docid", "faiss-docid"]
        self.faiss_index = None
        self.use_gpu = use_gpu
        self.single_gpu = faiss.StandardGpuResources() if use_gpu else None
        self.dim_size = 0
        self.results = {}
        self.mapping = {}
        self.rev_mapping = {}
    
    def _create_mapping_ids(self, corpus_ids):
        if not all(isinstance(doc_id, int) for doc_id in corpus_ids):
            for idx in range(len(corpus_ids)):
                self.mapping[corpus_ids[idx]] = idx
                self.rev_mapping[idx] = corpus_ids[idx]
    
    def _load(self, input_dir: str, prefix: str, ext: str):

        # Load ID mappings from file
        input_mappings_path = os.path.join(input_dir, "{}.{}.tsv".format(prefix, ext))
        print("Loading Faiss ID-mappings from path: {}".format(input_mappings_path))
        self.mapping = load_tsv_to_dict(input_mappings_path, header=True)
        self.rev_mapping = {v: k for k, v in self.mapping.items()}
        passage_ids = sorted(list(self.rev_mapping))
        
        # Load Faiss Index from disk
        input_faiss_path = os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))
        print("Loading Faiss Index from path: {}".format(input_faiss_path))
        
        return input_faiss_path, passage_ids

    def save(self, output_dir: str, prefix: str, ext: str):
        
        # Save BEIR -> Faiss ids mappings
        save_mappings_path = os.path.join(output_dir, "{}.{}.tsv".format(prefix, ext))
        print("Saving Faiss ID-mappings to path: {}".format(save_mappings_path))
        save_dict_to_tsv(self.mapping, save_mappings_path, keys=self.mapping_tsv_keys)

        # Save Faiss Index to disk
        save_faiss_path = os.path.join(output_dir, "{}.{}.faiss".format(prefix, ext))
        print("Saving Faiss Index to path: {}".format(save_faiss_path))
        self.faiss_index.save(save_faiss_path)
        print("Index size: {:.2f}MB".format(os.path.getsize(save_faiss_path)*0.000001))
    
    def _index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None):
        print("yes its here !")
        
        if(isinstance(self.corpus_embeddings,np.ndarray) and isinstance(self.faiss_ids,list)):
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            self._create_mapping_ids(corpus_ids)
            return self.faiss_ids, self.corpus_embeddings
        else:
            tqdm.write("creating index ...")
            #check
            print("Sorting Corpus by document length (Longest first)...")
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            self._create_mapping_ids(corpus_ids)
            corpus = [corpus[cid] for cid in corpus_ids]
            normalize_embeddings = True if score_function == "cos_sim" else False

            print("Encoding Corpus in batches... Warning: This might take a while!")

            itr = range(0, len(corpus), self.corpus_chunk_size)

            for batch_num, corpus_start_idx in enumerate(itr):
                print("Encoding Batch {}/{}. Normalize: {}...".format(batch_num+1, len(itr), normalize_embeddings))
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
                
                #Encode chunk of corpus    
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=True, 
                    normalize_embeddings=normalize_embeddings)
                
                if not batch_num: 
                    corpus_embeddings = sub_corpus_embeddings
                else:
                    corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])
            
            #Index chunk of corpus into faiss index
            print("Indexing Passages into Faiss...") 
            
            faiss_ids = [self.mapping.get(corpus_id) for corpus_id in corpus_ids]
            self.dim_size = corpus_embeddings.shape[1]

            del sub_corpus_embeddings

            return faiss_ids, corpus_embeddings
    
    def create_embeddings(self, queries: Dict[str, str],score_function = str):
        
        assert score_function in self.score_functions
        normalize_embeddings = True if score_function == "cos_sim" else False
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in queries]
        print("Computing Query Embeddings. Normalize: {}...".format(normalize_embeddings))
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=True, 
            batch_size=self.batch_size, 
            normalize_embeddings=normalize_embeddings)
        
        return query_ids, query_embeddings

    def search(self, 
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str], 
               top_k: int,
               score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:

        if not self.faiss_index: self.index(corpus, score_function)

        if(isinstance(self.query_embeddings,np.ndarray)==False):
            print("creating query embeddings")
            self.query_ids, self.query_embeddings = self.create_embeddings(queries=queries, score_function=score_function)
        
        print("searching..")
        faiss_scores, faiss_doc_ids = self.faiss_index.search(self.query_embeddings, top_k, **kwargs)

        #average_time_taken = round((time/len(self.query_embeddings)),3)
        
        #print(average_time_taken)
        
        for idx in range(len(self.query_ids)):
            scores = [float(score) for score in faiss_scores[idx]]
            if len(self.rev_mapping) != 0:
                doc_ids = [self.rev_mapping[doc_id] for doc_id in faiss_doc_ids[idx]]
            else:
                doc_ids = [str(doc_id) for doc_id in faiss_doc_ids[idx]]
            self.results[self.query_ids[idx]] = dict(zip(doc_ids, scores))
        
        return self.results


class BinaryFaissSearch(DenseRetrievalFaissSearch):

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "bin"):
        passage_embeddings = []
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index_binary(input_faiss_path)
        print("Reconstructing passage_embeddings back in Memory from Index...")
        for idx in tqdm(range(0, len(passage_ids)), total=len(passage_ids)):
            passage_embeddings.append(base_index.reconstruct(idx))            
        passage_embeddings = np.vstack(passage_embeddings)
        self.faiss_index = FaissBinaryIndex(base_index, passage_ids, passage_embeddings)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function)
        print("Using Binary Hashing in Flat Mode!")
        print("Output Dimension: {}".format(self.dim_size))
        base_index = faiss.IndexBinaryFlat(self.dim_size * 8)
        self.faiss_index = FaissBinaryIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "bin"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:

        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "binary_faiss_index"
    

class PQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size: int = 128, corpus_chunk_size: int = 50000, num_of_centroids: int = 96, 
                 code_size: int = 8, similarity_metric=faiss.METRIC_INNER_PRODUCT, use_rotation: bool = False, **kwargs):
        super(PQFaissSearch, self).__init__(model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size, corpus_chunk_size, **kwargs)
        self.num_of_centroids = num_of_centroids
        self.code_size = code_size
        self.similarity_metric = similarity_metric
        self.use_rotation = use_rotation
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "pq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissTrainIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissTrainIndex(base_index, passage_ids)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)  

        print("Using Product Quantization (PQ) in Flat mode!")
        print("Parameters Used: num_of_centroids: {} ".format(self.num_of_centroids))
        print("Parameters Used: code_size: {}".format(self.code_size))          
        
        base_index = faiss.IndexPQ(self.dim_size, self.num_of_centroids, self.code_size, self.similarity_metric)
        print("base index done...")
        if self.use_rotation:
            print("Rotating data before encoding it with a product quantizer...")
            print("Creating OPQ Matrix...")
            print("Input Dimension: {}, Output Dimension: {}".format(self.dim_size, self.num_of_centroids*4))
            opq_matrix = faiss.OPQMatrix(self.dim_size, self.code_size, self.num_of_centroids*4)
            base_index = faiss.IndexPQ(self.num_of_centroids*4, self.num_of_centroids, self.code_size, self.similarity_metric)
            base_index = faiss.IndexPreTransform(opq_matrix, base_index)
        
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            print("moved to gpu, now it should be built...")
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        
        else:
            print("building index...")
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "pq"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "pq_faiss_index"


class HNSWFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size: int = 128, corpus_chunk_size: int = 50000, hnsw_store_n: int = 512, 
                 hnsw_ef_search: int = 128, hnsw_ef_construction: int = 200, similarity_metric=faiss.METRIC_INNER_PRODUCT, **kwargs):
        super(HNSWFaissSearch, self).__init__(model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size, corpus_chunk_size, **kwargs)
        #super().__init__(model,batch_size)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.similarity_metric = similarity_metric

    tqdm.write("hnsw initialized..")
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "hnsw"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissHNSWIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissHNSWIndex(base_index, passage_ids)

    
    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)

        print("Using Approximate Nearest Neighbours (HNSW) in Flat Mode!")
        print("Parameters Required: hnsw_store_n: {}".format(self.hnsw_store_n))
        print("Parameters Required: hnsw_ef_search: {}".format(self.hnsw_ef_search))
        print("Parameters Required: hnsw_ef_construction: {}".format(self.hnsw_ef_construction))
        
        base_index = faiss.IndexHNSWFlat(self.dim_size + 1, self.hnsw_store_n, self.similarity_metric)
        base_index.hnsw.efSearch = self.hnsw_ef_search
        base_index.hnsw.efConstruction = self.hnsw_ef_construction
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissHNSWIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            print("preparing to build index")
            self.faiss_index = FaissHNSWIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "hnsw"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "hnsw_faiss_index"

class HNSWSQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size: int = 128, corpus_chunk_size: int = 50000, hnsw_store_n: int = 128, 
                 hnsw_ef_search: int = 128, hnsw_ef_construction: int = 200, similarity_metric=faiss.METRIC_INNER_PRODUCT, 
                 quantizer_type: str = "QT_8bit", **kwargs):
        super(HNSWSQFaissSearch, self).__init__(model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size, corpus_chunk_size, **kwargs)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.similarity_metric = similarity_metric
        self.qname = quantizer_type
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "hnsw-sq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        self.faiss_index = FaissTrainIndex(base_index, passage_ids)
    
    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)

        print("Using Approximate Nearest Neighbours (HNSW) in SQ Mode!")
        print("Parameters Required: hnsw_store_n: {}".format(self.hnsw_store_n))
        print("Parameters Required: hnsw_ef_search: {}".format(self.hnsw_ef_search))
        print("Parameters Required: hnsw_ef_construction: {}".format(self.hnsw_ef_construction))
        print("Parameters Required: quantizer_type: {}".format(self.qname))
        
        qtype = getattr(faiss.ScalarQuantizer, self.qname)
        base_index = faiss.IndexHNSWSQ(self.dim_size + 1, qtype, self.hnsw_store_n)
        base_index.hnsw.efSearch = self.hnsw_ef_search
        base_index.hnsw.efConstruction = self.hnsw_ef_construction
        self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "hnsw-sq"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "hnswsq_faiss_index"

class FlatIPFaissSearch(DenseRetrievalFaissSearch):
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "flat"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissIndex(base_index, passage_ids)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)
        base_index = faiss.IndexFlatIP(self.dim_size)
        #print("gpu index being created ...")
        #base_index = faiss.GpuIndexIVF.add_with_ids(x=corpus_embeddings,ids=faiss_ids)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "flat"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "flat_faiss_index"

class PCAFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, base_index: faiss.Index, output_dimension: int, batch_size: int = 128, 
                corpus_chunk_size: int = 50000, pca_matrix = None, random_rotation: bool = False, 
                eigen_power: float = 0.0, **kwargs):
        super(PCAFaissSearch, self).__init__(model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size, corpus_chunk_size, **kwargs)
        self.base_index = base_index
        self.output_dim = output_dimension
        self.pca_matrix = pca_matrix
        self.random_rotation = random_rotation
        self.eigen_power = eigen_power

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "pca"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissTrainIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissTrainIndex(base_index, passage_ids)


    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)
        print("Creating PCA Matrix...")
        print("Input Dimension: {}, Output Dimension: {}".format(self.dim_size, self.output_dim))
        pca_matrix = faiss.PCAMatrix(self.dim_size, self.output_dim, self.eigen_power, self.random_rotation)
        print("Random Rotation in PCA Matrix is set to: {}".format(self.random_rotation))
        print("Whitening in PCA Matrix is set to: {}".format(self.eigen_power))
        if self.pca_matrix is not None:
            pca_matrix = pca_matrix.copy_from(self.pca_matrix)
        self.pca_matrix = pca_matrix
        
        # Final index
        final_index = faiss.IndexPreTransform(pca_matrix, self.base_index)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, final_index)
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, final_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "pca"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "pca_faiss_index"

class SQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size: int = 128, corpus_chunk_size: int = 50000, 
                similarity_metric=faiss.METRIC_INNER_PRODUCT, quantizer_type: str = "QT_fp16", **kwargs):
        super(SQFaissSearch, self).__init__(model,corpus_embeddings,faiss_ids,query_ids, query_embeddings, batch_size, corpus_chunk_size, **kwargs)
        self.similarity_metric = similarity_metric
        self.qname = quantizer_type

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "sq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissTrainIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissTrainIndex(base_index, passage_ids)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)

        print("Using Scalar Quantizer in Flat Mode!")
        print("Parameters Used: quantizer_type: {}".format(self.qname))

        qtype = getattr(faiss.ScalarQuantizer, self.qname)
        base_index = faiss.IndexScalarQuantizer(self.dim_size, qtype, self.similarity_metric)
        if self.use_gpu:
            print("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "sq"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)
    
    def get_index_name(self):
        return "sq_faiss_index"