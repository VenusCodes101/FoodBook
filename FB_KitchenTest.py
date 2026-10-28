#FoodBook test 
import numpy as np
import faiss
import time 

embedding_dim = 128 
num_clusters = 100

#training data 
data_vectors = np.random.random((10000, embedding_dim)).astype('float32')

#building the IVF-PQ [Inverted File with Product Quantization]

#idk exactly what is happening right here soooo
quantizer = faiss.IndexFlatL2(embedding_dim)

#16 is the number of subquantizers(???) 8 is the bits per subvector
index = faiss.IndexIVFPQ(quantizer, embedding_dim, num_clusters, 16, 8)

#Train the index that is required for IVF-PQ
index.train(data_vectors)

#Adding vectors
index.add(data_vectors)

#number of clusters to probe during search (???)
index.nprobe = 10 

#Query Vector 
query_vector = np.random.random((1, embedding_dim)).astype('float32')
k = 5

#timing the search
start_time = time.time()
distances, indices = index.search(query_vector , k)
end_time = time.time()

elapsed_ms = (end_time - start_time)*1000

#final words
print("Nearest neighbor indices: ", indices)
print("Distances: ", distances)
print(f"Search time (IVF-PQ ANN): {elapsed_ms:.4f} ms")