import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import Levenshtein
import time
import cologne_phonetics
import jellyfish

def main():
    st.title('K-nearest neighbor')
    
    # Start measuring execution time
    start_time = time.time()
    
    # Read the CSV data
    data = pd.read_csv('names1.csv',dtype=str)
    data['name'] = data['name'].str.lower()
    print(data["cologne"].iloc[0])
    # Set up the layout to display the 'name' column table on the right side
    col1, col2 = st.columns([2, 1])  # 2/3 of the page width for col1, 1/3 for col2
    
    # Show the 'name' column in a table with a width of 500 pixels on the right side
    with col2:
        st.write('Data Name')
        st.dataframe(data['name'], width=500)  # Replace 'name' with the actual column name in your CSV
    
    # Show some statistics on the left side (col1)
    with col1:
        st.write('Find the 5 Nearest Neighbors:')
        new_name = st.text_input('Enter a name to find its neighbors:')
        new_name_space = new_name.replace(' ','')
        new_name = new_name.lower()

        if new_name_space:
            # Your existing code for creating the KNN model and finding neighbors
            name_ngram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            metaphone_ngram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            cologne_ngram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            
            
            name_vectors_ngram = name_ngram_vectorizer.fit_transform(data['name'])
            metaphone_vectors = metaphone_ngram_vectorizer.fit_transform(data['metaphone'])
            cologne_vectors = cologne_ngram_vectorizer.fit_transform(data['cologne'].astype(str))
            
            
            combined_feature_vectors = np.concatenate((metaphone_vectors.toarray(), cologne_vectors.toarray(),name_vectors_ngram.toarray()) , axis=1)
            k = 5  # Number of neighbors to consider
            knn = NearestNeighbors(n_neighbors=k, metric='jaccard', algorithm='brute')
            knn.fit(combined_feature_vectors)

            # Transform user input name and find neighbors
            new_name_vector_ngram = name_ngram_vectorizer.transform([new_name_space])
            new_metaphone_vector = metaphone_ngram_vectorizer.transform([jellyfish.metaphone(new_name_space)])
            new_cologne_vector = cologne_ngram_vectorizer.transform([cologne_phonetics.encode(new_name_space)[0][1]])
            new_combined_feature_vector = np.concatenate((new_metaphone_vector.toarray(), new_cologne_vector.toarray(), new_name_vector_ngram.toarray()), axis=1)
            distances, indices = knn.kneighbors(new_combined_feature_vector)
            
            # Get the names of the K-nearest neighbors
            nearest_names = data['name'].iloc[indices[0]]
            print(nearest_names)
           # Display the K-nearest neighbor names
            matrice_name = []
            #print(nearest_names)
            for i,name in enumerate(nearest_names):
                metaphone_simirality_score = round(Levenshtein.ratio(jellyfish.metaphone(new_name), data['metaphone'].iloc[indices[0][i]]) * 100, 2)
                cologne_similarity_score = round(Levenshtein.ratio(cologne_phonetics.encode(new_name_space)[0][1], data['cologne'].iloc[indices[0][i]]) * 100, 2)
                name_similarity_score = round(Levenshtein.ratio(new_name,name)*100,2)

                
                print("similarity between", new_name_space, "and ",name," => cologone = ",cologne_similarity_score, "\\" , " metaphone = ",metaphone_simirality_score )
                
                print("metaphone ->", new_name," = ",jellyfish.metaphone(new_name).replace(" ",""), " ", name, " = " ,data['metaphone'].iloc[indices[0][i]])
                print("cologone ->", new_name," = ",cologne_phonetics.encode(new_name_space)[0][1], " ", name, " = " ,data['cologne'].iloc[indices[0][i]])
                
                
                similarity_score =((metaphone_simirality_score + cologne_similarity_score + name_similarity_score) / 3) 
               
                x = [name, similarity_score]
                matrice_name.append(x)

            matrice_name.sort(key=lambda x: x[1], reverse=True)
            
            # End measuring execution time
            end_time = time.time()
            execution_time = end_time - start_time

           # Display execution time with bold formatting
            st.write(f"<span style='font-weight: bold'>Execution time: {execution_time:.2f} seconds</span>", unsafe_allow_html=True)
            st.write(f"The 5-nearest neighbors of '{new_name}' are:")
            for i, name in enumerate(nearest_names):
                score = matrice_name[i][1]
                if score >= 80.0:
                    st.write(f"{i+1}. <span style='color:green'>{matrice_name[i][0]} % <span style='color:green'>{score}</span>", unsafe_allow_html=True)
                else:
                    st.write(f"{i+1}. <span style='color:red'>{matrice_name[i][0]} % <span style='color:red'>{score}</span>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
