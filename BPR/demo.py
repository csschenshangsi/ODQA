import faiss
from bpr import BiEncoder, FaissBinaryIndex, InMemoryPassageDB, Retriever
# Load the model from the checkpoint file
biencoder = BiEncoder.load_from_checkpoint("bpr_finetuned_nq.ckpt")
biencoder.eval()
biencoder.freeze()
# Load Wikipedia passages into memory
passage_db = InMemoryPassageDB("psgs_w100.tsv")
# Load the index
base_index = faiss.read_index_binary("bpr_finetuned_nq.idx")  # 这个index 文件是哪里来的？
index = FaissBinaryIndex(base_index)
# Instantiate the Retriever
retriever = Retriever(index, biencoder, passage_db)
# Encode queries into embeddings
query_embeddings = retriever.encode_queries(["what is the tallest mountain in the world"])
# Get top-100 results
Candidate = retriever.search(query_embeddings, k=100)[0][0]
#Candidate(
# id=525407,
# score=93.59397888183594,
# passage=Passage(
# id=525407,
# title='Mount Everest',
# text="Mount Everest Mount Everest, known in Nepali as Sagarmatha () and in Tibetan as Chomolungma (), is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The international border between Nepal (Province No. 1) and China (Tibet Autonomous Region) runs across its summit point. The current official elevation of , recognized by China and Nepal, was established by a 1955 Indian survey and subsequently confirmed by a Chinese survey in 1975. In 2005, China remeasured the rock height of the mountain, with a result of 8844.43 m. There followed an argument between China and"))
#The Wikipedia passage data (psgs_w100.tsv) is available on the DPR website. At the time of writing, the file can be downloaded by cloning the DPR repository and running the following command: