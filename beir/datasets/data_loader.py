from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
import pickle

logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = "", limit: int=50000):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.limit = limit

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file
        print("initialised")

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            print("Loading Corpus...")
            self._load_corpus()
            print("Loaded %d Documents.", len(self.corpus))
            print("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            print("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            print("Loaded %d Queries.", len(self.queries))
            print("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        # try:
        #     self.check(fIn=self.corpus_file, ext="jsonl")
        # except Exception:
        #     self.check(fIn=self.corpus_file, ext="pickle")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            #print("Loading Corpus...x")
            self._load_corpus()
            # print("Loaded %d %s Documents.", len(self.corpus), split.upper())
            # print("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            print("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            print("Loading qrels...")
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            # print("Loaded %d %s Queries.", len(self.queries), split.upper())
            # print("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        if(self.check(fIn=self.corpus_file, ext="jsonl")):
            if not len(self.corpus):
                print("Loading Corpus...")
                self._load_corpus()
                # print("Loaded %d Documents.", len(self.corpus))
                # print("Doc Example: %s", list(self.corpus.values())[0])

            return self.corpus
        elif(self.check(fIn=self.corpus_file, ext="pickle")):
            with open(self.corpus_file, 'rb', encoding='utf8') as file:
                self.corpus = pickle.load(file)
            return self.corpus
    
    def _load_corpus(self):
        if("jsonl" in self.corpus_file):
            num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
            #print("loading limited lines:",self.limit)
            with open(self.corpus_file, encoding='utf8') as fIn:
                for line in tqdm(fIn, total=num_lines):
                    line = json.loads(line)
                    self.corpus[line.get("_id")] = {
                        "text": line.get("text"),
                        "title": line.get("title"),
                    }
                # for line in fIn.readlines(self.limit):
                #     line = json.loads(line)
                #     self.corpus[line.get("_id")] = {"text": line.get("text"),"title": line.get("title")}
        elif("pickle" in self.corpus_file):
            with open(self.corpus_file, 'rb', encoding='utf8') as file:
                self.corpus = pickle.load(file)
            
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        print("opening the custom file..")
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score