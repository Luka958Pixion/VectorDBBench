from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

class ChromaConfig(DBConfig):
    password: SecretStr
    host: SecretStr
    port: int 

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value(),
        }
        
class ChromaIndexConfig(BaseModel):
    """Base config for chroma"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value
    
class ChromaHNSWConfig(ChromaIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "construction_ef": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"search_ef": self.ef},
        }