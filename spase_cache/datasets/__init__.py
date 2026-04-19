from spase_cache.datasets.base import Dataset
from spase_cache.datasets.sharegpt import ShareGPTDataset
from spase_cache.datasets.revisions import (
    RevisionDataset,
    WikipediaDataset,
    GitHubDataset,
    GitProjectDataset,
)
from spase_cache.datasets.single_doc_qa import (
    SingleDocQADataset,
    SquadDataset,
    QuALITYDataset,
    MuLDDataset,
)
from spase_cache.datasets.agentic import (
    AgenticDataset,
    SweAgentDataset,
    NemotronSweDataset,
)
from spase_cache.datasets.tree_of_thoughts import (
    TreeDataset,
    Osst1Dataset,
)
from spase_cache.datasets.synthetic import SyntheticDataset
