from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Dict, Union, Tuple



class langchain_hft:

    def __init__(self, model_name: Union[str, Tuple] = None, device: str = 'cpu', normalize_embeddings: bool = True, sep: str = " ", **kwargs):
        self.sep = sep
        self.normalize_embeddings = normalize_embeddings
        self.device = device

        if isinstance(model_name, str):
            self.model_name = model_name
        
        if(self.device in ['cpu','cuda']):
            model_kwargs = {'device': self.device}

        if(self.normalize_embeddings):
            encode_kwargs = {'normalize_embeddings': True}

        self.model = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs):

        