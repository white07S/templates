import lunr from 'lunr';

let searchData = null;

// Load the pre-built search index
export const loadSearchIndex = async () => {
  if (searchData) {
    return searchData;
  }

  try {
    const response = await fetch('/searchIndex.json');
    if (!response.ok) {
      throw new Error('Failed to load search index');
    }

    const data = await response.json();

    // Rebuild the Lunr index from the serialized version
    const index = lunr.Index.load(data.index);

    searchData = {
      index,
      documents: data.documents
    };

    return searchData;
  } catch (error) {
    console.error('Error loading search index:', error);
    // Return empty data structure as fallback
    return {
      index: null,
      documents: {}
    };
  }
};

// Search function using the loaded index
export const searchDocuments = (query, data) => {
  if (!query || query.length < 2 || !data?.index) {
    return [];
  }

  try {
    // Use fuzzy search with wildcard
    const searchQuery = query
      .split(' ')
      .map(term => `+${term}* ${term}~1`)
      .join(' ');

    const results = data.index.search(searchQuery);

    return results
      .map(result => {
        const doc = data.documents[result.ref];
        return {
          ...result,
          document: doc
        };
      })
      .slice(0, 10); // Return top 10 results
  } catch (error) {
    console.error('Search error:', error);
    return [];
  }
};