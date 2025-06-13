import numpy as np
import pandas as pd
import hdbscan

def perform_concentration_analysis(df, start_date, end_date):
    # Step 1: Filter by date
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    filtered_df = df.loc[mask]
    
    # Step 2: Flatten data
    data = []
    for _, row in filtered_df.iterrows():
        loss_id = row['Loss_id']
        loss_desc = row['Loss_description']
        root_causes = row['loss_root_causes']
        embeddings_list = row['loss_root_causes_embeddings']
        for i, sentence in enumerate(root_causes):
            data.append({
                'loss_id': loss_id,
                'loss_description': loss_desc,
                'sentence': sentence,
                'embedding': embeddings_list[i]
            })
    
    if not data:
        return []
    
    # Step 3: Cluster embeddings
    embeddings_matrix = np.vstack([d['embedding'] for d in data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
    cluster_labels = clusterer.fit_predict(embeddings_matrix)
    
    # Handle noise (assign unique cluster labels)
    max_label = max(cluster_labels) if cluster_labels.size > 0 else 0
    new_labels = np.where(cluster_labels == -1, [max_label + 1 + i for i in range(len(cluster_labels))], cluster_labels)
    
    # Step 4: Group into clusters
    clusters = {}
    for idx, label in enumerate(new_labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = {
                'sentences': [],
                'embeddings': [],
                'loss_ids': set(),
                'loss_descriptions': {}
            }
        clusters[label]['sentences'].append(data[idx]['sentence'])
        clusters[label]['embeddings'].append(data[idx]['embedding'])
        clusters[label]['loss_ids'].add(data[idx]['loss_id'])
        clusters[label]['loss_descriptions'][data[idx]['loss_id']] = data[idx]['loss_description']
    
    # Step 5: Process clusters
    results = []
    for label, cluster_data in clusters.items():
        embeds = np.array(cluster_data['embeddings'])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        normalized_embeds = embeds / norms
        
        # Compute centroid and find representative sentence
        centroid = np.mean(normalized_embeds, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        cosine_similarities = np.dot(normalized_embeds, centroid_norm)
        rep_idx = np.argmax(cosine_similarities)
        representative_sentence = cluster_data['sentences'][rep_idx]
        
        # Compile losses
        losses_list = [{
            'loss_id': loss_id, 
            'loss_description': cluster_data['loss_descriptions'][loss_id]
        } for loss_id in cluster_data['loss_ids']]
        
        # Append cluster result
        results.append({
            'root_cause_sentence': representative_sentence,
            'rank_score': len(cluster_data['loss_ids']),  # Unique loss IDs count
            'losses': losses_list
        })
    
    # Step 6: Sort by rank_score (descending)
    results.sort(key=lambda x: x['rank_score'], reverse=True)
    return results

# Example Usage
# result = perform_concentration_analysis(df, '2023-01-01', '2023-12-31')
