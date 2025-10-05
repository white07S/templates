"""Business logic for documentation system"""
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import tantivy

# Set up logging
logger = logging.getLogger(__name__)

# Constants
CURRENT_SCHEMA_VERSION = "2.1.0"

class DocumentationService:
    """Service class for managing documentation indexing and search"""

    def __init__(self, docs_dir: Path, index_dir: Path):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(exist_ok=True)
        self.images_dir = self.docs_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.index_dir = Path(index_dir)
        self.schema_version_file = self.index_dir / "schema_version.json"

        self.index = None
        self.searcher = None

    def create_schema(self):
        """Create the Tantivy schema"""
        schema_builder = tantivy.SchemaBuilder()

        # Document fields with proper configuration
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")  # Use raw tokenizer for IDs
        schema_builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
        schema_builder.add_text_field("content", stored=True, tokenizer_name="en_stem")
        schema_builder.add_text_field("path", stored=True, tokenizer_name="raw")  # Use raw tokenizer for paths
        schema_builder.add_text_field("category", stored=True, tokenizer_name="raw")  # Categories don't need stemming
        schema_builder.add_text_field("heading", stored=True, tokenizer_name="en_stem")
        schema_builder.add_text_field("anchor", stored=True, tokenizer_name="raw")  # Anchors are exact strings
        schema_builder.add_text_field("doc_title", stored=True, tokenizer_name="en_stem")
        # Add fields for prefix search using default tokenizer (which supports prefix queries)
        schema_builder.add_text_field("title_prefix", stored=False, tokenizer_name="default")
        schema_builder.add_text_field("content_prefix", stored=False, tokenizer_name="default")

        return schema_builder.build()

    def check_schema_compatibility(self) -> bool:
        """Check if the existing index schema is compatible"""
        if not self.index_dir.exists():
            return False

        if not self.schema_version_file.exists():
            return False

        try:
            with open(self.schema_version_file, 'r') as f:
                version_info = json.load(f)
                return version_info.get('version') == CURRENT_SCHEMA_VERSION
        except:
            return False

    def save_schema_version(self):
        """Save the current schema version"""
        self.index_dir.mkdir(exist_ok=True)
        with open(self.schema_version_file, 'w') as f:
            json.dump({
                'version': CURRENT_SCHEMA_VERSION,
                'fields': [
                    'id', 'title', 'content', 'path',
                    'category', 'heading', 'anchor', 'doc_title',
                    'title_prefix', 'content_prefix'
                ]
            }, f, indent=2)

    def initialize_or_recreate_index(self):
        """Initialize the index, recreating if schema has changed"""
        schema = self.create_schema()

        # Check if we need to recreate the index
        if self.index_dir.exists():
            if not self.check_schema_compatibility():
                logger.info("Schema version mismatch. Recreating index...")
                shutil.rmtree(self.index_dir)
                self.index_dir.mkdir(exist_ok=True)
            else:
                try:
                    # Try to open existing index
                    self.index = tantivy.Index(str(self.index_dir))
                    self.searcher = self.index.searcher()
                    logger.info("Loaded existing Tantivy index")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}. Recreating...")
                    shutil.rmtree(self.index_dir)
                    self.index_dir.mkdir(exist_ok=True)
        else:
            self.index_dir.mkdir(exist_ok=True)

        # Create new index
        self.index = tantivy.Index(schema, str(self.index_dir))
        self.searcher = self.index.searcher()
        self.save_schema_version()
        logger.info("Created new Tantivy index")

    def extract_sections_from_mdx(self, content: str, doc_id: str, path: str) -> List[Dict[str, str]]:
        """Extract different sections from MDX content for granular indexing"""
        sections = []
        lines = content.split("\n")
        current_heading = None
        current_content = []
        current_heading_level = 0

        # Extract document title
        doc_title = path.replace(".mdx", "").replace("-", " ").title()
        for line in lines:
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break

        for i, line in enumerate(lines):
            # Check if it's a heading
            if line.startswith("#"):
                # Save previous section if exists
                if current_content and current_heading:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        # Create anchor from heading
                        anchor = current_heading.lower().replace(" ", "-").replace(".", "")
                        sections.append({
                            "id": f"{doc_id}#{anchor}",
                            "title": doc_title,
                            "content": content_text,
                            "heading": current_heading,
                            "category": "content",
                            "anchor": anchor,
                            "path": path,
                            "doc_title": doc_title
                        })

                # Extract heading level and text
                heading_match = line.split(" ", 1)
                if len(heading_match) > 1:
                    current_heading_level = len(heading_match[0])
                    current_heading = heading_match[1].strip()
                    current_content = []

                    # Index the heading itself
                    anchor = current_heading.lower().replace(" ", "-").replace(".", "")
                    sections.append({
                        "id": f"{doc_id}#{anchor}-heading",
                        "title": doc_title,
                        "content": current_heading,
                        "heading": current_heading,
                        "category": "heading",
                        "anchor": anchor,
                        "path": path,
                        "doc_title": doc_title
                    })

            # Check if it's a code block
            elif line.startswith("```"):
                code_lines = []
                j = i + 1
                while j < len(lines) and not lines[j].startswith("```"):
                    code_lines.append(lines[j])
                    j += 1

                if code_lines:
                    code_content = "\n".join(code_lines)
                    # Create a stable anchor for code blocks
                    code_hash = hashlib.md5(code_content.encode()).hexdigest()[:8]
                    anchor = f"code-{code_hash}"
                    sections.append({
                        "id": f"{doc_id}#{anchor}",
                        "title": doc_title,
                        "content": code_content,
                        "heading": current_heading or "",
                        "category": "code",
                        "anchor": anchor,
                        "path": path,
                        "doc_title": doc_title
                    })
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content and current_heading:
            content_text = "\n".join(current_content).strip()
            if content_text:
                anchor = current_heading.lower().replace(" ", "-").replace(".", "")
                sections.append({
                    "id": f"{doc_id}#{anchor}",
                    "title": doc_title,
                    "content": content_text,
                    "heading": current_heading,
                    "category": "content",
                    "anchor": anchor,
                    "path": path,
                    "doc_title": doc_title
                })

        return sections

    def index_documents(self):
        """Index all MDX documents for search with granular sections"""
        if not self.index:
            self.initialize_or_recreate_index()

        writer = self.index.writer()

        # Clear existing documents
        writer.delete_all_documents()

        document_count = 0

        # Index all MDX files
        for mdx_file in self.docs_dir.glob("**/*.mdx"):
            relative_path = str(mdx_file.relative_to(self.docs_dir))
            doc_id = relative_path.replace("/", "_").replace(".mdx", "")

            try:
                with open(mdx_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract sections for granular indexing
                sections = self.extract_sections_from_mdx(content, doc_id, relative_path)

                # Index each section separately
                for section in sections:
                    # Combine heading and content for better search
                    search_content = section["content"]
                    if section.get("heading") and section["category"] != "heading":
                        search_content = f"{section['heading']} {search_content}"

                    doc = tantivy.Document()
                    doc.add_text("id", section["id"])
                    doc.add_text("title", section["title"])
                    doc.add_text("content", search_content)
                    doc.add_text("path", section["path"])
                    doc.add_text("category", section["category"])
                    doc.add_text("heading", section.get("heading", ""))
                    doc.add_text("anchor", section.get("anchor", ""))
                    doc.add_text("doc_title", section["doc_title"])
                    # Add prefix search fields
                    doc.add_text("title_prefix", section["title"])
                    doc.add_text("content_prefix", search_content)

                    writer.add_document(doc)
                    document_count += 1

            except Exception as e:
                logger.error(f"Error indexing {mdx_file}: {e}")
                continue

        writer.commit()
        self.index.reload()
        self.searcher = self.index.searcher()

        logger.info(f"Successfully indexed {document_count} document sections")

    def get_navigation_structure(self) -> List[Dict[str, Any]]:
        """Build hierarchical navigation structure from file system"""
        nav_structure = {}

        # Get all MDX files
        mdx_files = list(self.docs_dir.glob("**/*.mdx"))

        for mdx_file in sorted(mdx_files):
            relative_path = mdx_file.relative_to(self.docs_dir)
            parts = relative_path.parts

            # Extract title from file content
            with open(mdx_file, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                title = mdx_file.stem.replace("-", " ").title()
                for line in lines:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

            doc_id = str(relative_path).replace("/", "_").replace(".mdx", "")

            # Build hierarchical structure
            current_level = nav_structure

            # Process each directory level
            for i, part in enumerate(parts[:-1]):  # All parts except the file
                # Create directory entry if it doesn't exist
                if part not in current_level:
                    dir_id = "_".join(parts[:i+1])
                    dir_title = part.replace("-", " ").title()

                    # Check if there's an index.mdx file for this directory
                    index_file = self.docs_dir / "/".join(parts[:i+1]) / "index.mdx"
                    if index_file.exists():
                        with open(index_file, "r", encoding="utf-8") as f:
                            index_content = f.read()
                            for line in index_content.split("\n"):
                                if line.startswith("# "):
                                    dir_title = line[2:].strip()
                                    break

                    current_level[part] = {
                        "id": dir_id,
                        "title": dir_title,
                        "path": "/".join(parts[:i+1]),
                        "is_directory": True,
                        "children": {}
                    }

                current_level = current_level[part]["children"]

            # Add the file to the appropriate level
            file_name = parts[-1]

            # Skip index.mdx files as they represent the parent directory
            if file_name == "index.mdx":
                # Update parent directory info with index content
                parent_parts = parts[:-1]
                if parent_parts:
                    parent_level = nav_structure
                    for part in parent_parts[:-1]:
                        parent_level = parent_level[part]["children"]
                    if parent_parts[-1] in parent_level:
                        parent_level[parent_parts[-1]]["path"] = str(relative_path)
                        parent_level[parent_parts[-1]]["title"] = title
                        parent_level[parent_parts[-1]]["has_index"] = True
                continue

            current_level[file_name] = {
                "id": doc_id,
                "title": title,
                "path": str(relative_path),
                "is_directory": False,
                "children": []
            }

        # Convert nested dict to list format
        def dict_to_list(items_dict):
            result = []
            for key, item in sorted(items_dict.items()):
                if item.get("is_directory"):
                    # Directory with children
                    children = dict_to_list(item["children"]) if item["children"] else []
                    result.append({
                        "id": item["id"],
                        "title": item["title"],
                        "path": item["path"],
                        "children": children,
                        "has_index": item.get("has_index", False)
                    })
                else:
                    # Regular file
                    result.append({
                        "id": item["id"],
                        "title": item["title"],
                        "path": item["path"],
                        "children": item.get("children", [])
                    })
            return result

        return dict_to_list(nav_structure)

    def get_page_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Get specific page content"""
        # Convert ID back to file path
        file_path = page_id.replace("_", "/") + ".mdx"
        full_path = self.docs_dir / file_path

        if not full_path.exists():
            return None

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract title
        lines = content.split("\n")
        title = full_path.stem.replace("-", " ").title()
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return {
            "id": page_id,
            "title": title,
            "content": content,
            "path": file_path
        }

    def search_documents(self, query: str, category: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search documents with optional category filtering"""
        if not self.searcher:
            raise Exception("Search index not initialized")

        query_trimmed = query.strip().lower()

        # Check if this is a partial word (doesn't end with space)
        is_partial = not query.endswith(' ') and len(query_trimmed) >= 2

        results = []

        # For partial queries, we'll do a broader search and then filter
        if is_partial:
            # Get ALL documents and filter them for partial matches
            # This is less efficient but ensures we get partial word matches
            all_docs_query = "*"  # Match all documents
            try:
                # Try to get a reasonable number of documents to search through
                query_parser = self.index.parse_query(all_docs_query, ["content"])
                search_results = self.searcher.search(query_parser, limit=1000)
            except:
                # If wildcard doesn't work, try empty query or fall back to regular search
                query_parser = self.index.parse_query(query_trimmed, ["title", "content", "heading", "doc_title"])
                search_results = self.searcher.search(query_parser, limit=limit * 10)

            # Manual filtering for partial matches
            for score, doc_address in search_results.hits:
                doc = self.searcher.doc(doc_address)

                # Get document fields
                title = (doc.get_first("title") or "").lower()
                content = (doc.get_first("content") or "").lower()
                heading = (doc.get_first("heading") or "").lower()
                doc_title = (doc.get_first("doc_title") or "").lower()

                # Check if query appears as substring in any field
                if (query_trimmed in title or
                    query_trimmed in content or
                    query_trimmed in heading or
                    query_trimmed in doc_title):

                    # Calculate a relevance score based on where the match was found
                    relevance_score = 0
                    if query_trimmed in title or query_trimmed in doc_title:
                        relevance_score += 10
                    if query_trimmed in heading:
                        relevance_score += 5
                    if query_trimmed in content:
                        relevance_score += 1

                    results.append((relevance_score, doc_address))

            # Sort by relevance score
            results.sort(key=lambda x: x[0], reverse=True)
            results = results[:limit * 2]

        else:
            # For complete word queries, use normal Tantivy search
            query_parser = self.index.parse_query(query_trimmed, ["title", "content", "heading", "doc_title"])
            search_results = self.searcher.search(query_parser, limit=limit * 2)
            results = [(score, doc_address) for score, doc_address in search_results.hits]

        # Process results
        final_results = []
        processed_ids = set()

        for score, doc_address in results:
            doc = self.searcher.doc(doc_address)

            doc_id = doc.get_first("id")
            if not doc_id:
                continue

            # Skip duplicates
            if doc_id in processed_ids:
                continue
            processed_ids.add(doc_id)

            # Get all fields with fallbacks
            title = doc.get_first("doc_title") or doc.get_first("title") or "Untitled"
            content = doc.get_first("content") or ""
            path = doc.get_first("path") or ""
            result_category = doc.get_first("category") or "content"
            heading = doc.get_first("heading") or ""
            anchor = doc.get_first("anchor") or ""

            # Filter by category if specified
            if category and result_category != category:
                continue

            # Create snippet
            snippet = self.create_snippet(content, query)

            final_results.append({
                "id": doc_id,
                "title": title,
                "path": path,
                "score": score,
                "snippet": snippet,
                "category": result_category,
                "heading": heading if heading else None,
                "anchor": anchor if anchor else None
            })

            # Stop if we have enough results
            if len(final_results) >= limit:
                break

        # Sort by score (highest first) - for partial matches, score is already relevance-based
        if not is_partial:
            final_results.sort(key=lambda x: x["score"], reverse=True)

        return final_results

    def create_snippet(self, content: str, query: str, max_length: int = 150) -> str:
        """Create a snippet with the query highlighted"""
        content_lower = content.lower()
        query_lower = query.lower()

        # Find the position of the query in the content
        pos = content_lower.find(query_lower)

        if pos == -1:
            # If exact match not found, try to find any of the words
            words = query_lower.split()
            for word in words:
                pos = content_lower.find(word)
                if pos != -1:
                    break

        if pos == -1:
            # Return beginning of content if no match
            return content[:max_length] + "..." if len(content) > max_length else content

        # Calculate snippet boundaries
        start = max(0, pos - 50)
        end = min(len(content), pos + len(query) + 100)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def get_navigation_context(self, page_id: str) -> Dict[str, Any]:
        """Get previous and next page for navigation in hierarchical structure"""
        pages = self.get_navigation_structure()

        # Flatten the hierarchical structure for navigation
        def flatten_pages(items, result=None):
            if result is None:
                result = []

            for item in items:
                # Add the item if it has content (not just a directory without index)
                if not item.get("is_directory") or item.get("has_index"):
                    result.append({
                        "id": item["id"],
                        "title": item["title"],
                        "path": item["path"]
                    })

                # Recursively add children
                if item.get("children"):
                    flatten_pages(item["children"], result)

            return result

        flat_pages = flatten_pages(pages)

        # Find current page index
        current_index = -1
        for i, page in enumerate(flat_pages):
            if page["id"] == page_id:
                current_index = i
                break

        if current_index == -1:
            return None

        previous_page = flat_pages[current_index - 1] if current_index > 0 else None
        next_page = flat_pages[current_index + 1] if current_index < len(flat_pages) - 1 else None

        return {
            "previous": previous_page,
            "next": next_page,
            "current": flat_pages[current_index]
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Check the health of the service and search index"""
        try:
            index_exists = self.index_dir.exists()
            schema_version = None
            doc_count = 0

            if self.schema_version_file.exists():
                with open(self.schema_version_file, 'r') as f:
                    version_info = json.load(f)
                    schema_version = version_info.get('version')

            if self.searcher:
                # Get document count (approximate)
                doc_count = len(list(self.docs_dir.glob("**/*.mdx")))

            return {
                "status": "healthy",
                "index_exists": index_exists,
                "schema_version": schema_version,
                "current_schema_version": CURRENT_SCHEMA_VERSION,
                "schema_compatible": schema_version == CURRENT_SCHEMA_VERSION,
                "approximate_documents": doc_count
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }