# sp_page_reconstruct.py
import os
import json
import html
import requests
from typing import Dict, Any, List, Optional

# ---------------------------
# Config (env vars preferred)
# ---------------------------
TENANT_ID = os.getenv("AZURE_TENANT_ID", "<tenant-guid-or-domain>")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "<app-id>")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")  # for app-only
USE_DEVICE_CODE = os.getenv("USE_DEVICE_CODE", "0") == "1"  # set 1 to use user sign-in

# Graph resource + scopes
GRAPH_SCOPE_DEFAULTS = ["https://graph.microsoft.com/.default"]  # for app-only
GRAPH_SCOPES_DELEGATED = ["Sites.Read.All", "Pages.Read.All", "offline_access", "openid", "profile"]

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SP_REST_BASE_FMT = "https://{host}/_api"  # used only for CanvasContent1 fallback

# ---------------------------
# Auth (MSAL)
# ---------------------------
def get_access_token() -> str:
    import msal

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"

    if USE_DEVICE_CODE:
        # Delegated flow (user sign-in) – requires app to have delegated permissions granted
        app = msal.PublicClientApplication(CLIENT_ID, authority=authority)
        flow = app.initiate_device_flow(scopes=[f"{s}" for s in GRAPH_SCOPES_DELEGATED])
        if "user_code" not in flow:
            raise RuntimeError(f"Failed to create device flow: {flow}")
        print(f"To sign in, visit {flow['verification_uri']} and enter code: {flow['user_code']}")
        result = app.acquire_token_by_device_flow(flow)
    else:
        # App-only (client credentials) – requires application permissions + admin consent
        if not CLIENT_SECRET:
            raise RuntimeError("CLIENT_SECRET missing for app-only auth.")
        app = msal.ConfidentialClientApplication(
            CLIENT_ID, authority=authority, client_credential=CLIENT_SECRET
        )
        result = app.acquire_token_for_client(scopes=GRAPH_SCOPE_DEFAULTS)

    if "access_token" not in result:
        raise RuntimeError(f"Auth failed: {result}")
    return result["access_token"]

# ---------------------------
# Graph helpers
# ---------------------------
def gget(token: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=60)
    if not r.ok:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text}")
    return r.json()

def get_page(token: str, site_id: str, page_id: str) -> Dict[str, Any]:
    # e.g., site_id: "contoso.sharepoint.com,12345-abc,6789-def"
    url = f"{GRAPH_BASE}/sites/{site_id}/pages/{page_id}"
    return gget(token, url)

def get_page_webparts(token: str, site_id: str, page_id: str) -> List[Dict[str, Any]]:
    url = f"{GRAPH_BASE}/sites/{site_id}/pages/{page_id}/microsoft.graph.sitePage/webparts"
    data = gget(token, url)
    return data.get("value", [])

# ---------------------------
# Simple renderer
# ---------------------------
def escape(s: str) -> str:
    return html.escape(s or "", quote=True)

def render_webpart(wp: Dict[str, Any]) -> str:
    wp_type = (wp.get("type") or "").lower()
    data = wp.get("data", {})
    # Text web part commonly exposes HTML in 'text' or 'innerHtml' (varies).
    if "textwebpart" in wp_type:
        inner = data.get("text") or data.get("innerHtml") or ""
        # 'inner' is usually already HTML – keep as-is (assume trusted source from SharePoint).
        return f'<div class="sp-webpart sp-text">{inner}</div>'

    # Image-like parts sometimes provide a 'imageSource' or 'file' reference; show a light card
    if "image" in wp_type and isinstance(data, dict):
        src = data.get("imageSource") or data.get("file") or ""
        caption = data.get("caption") or ""
        return f'''
        <figure class="sp-webpart sp-image">
          <div><em>{escape(str(src))}</em></div>
          <figcaption>{escape(caption)}</figcaption>
        </figure>
        '''

    # File viewer, QuickLinks, List, Hero, Embed, etc. – show a compact JSON inspector
    pretty = escape(json.dumps(data, ensure_ascii=False, indent=2))
    tlabel = escape(wp.get("type") or "WebPart")
    return f'''
    <div class="sp-webpart sp-unknown">
      <details open>
        <summary>{tlabel}</summary>
        <pre>{pretty}</pre>
      </details>
    </div>
    '''

def reconstruct_html(page: Dict[str, Any], webparts: List[Dict[str, Any]]) -> str:
    title = page.get("title") or page.get("name") or "Page"
    # Try to respect canvas layout if present; otherwise stack in order.
    layout = page.get("canvasLayout", {})
    sections = layout.get("horizontalSections") or []

    if not sections:
        # No layout info – just render all parts in order
        body = "\n".join(render_webpart(wp) for wp in webparts)
        return base_html(title, f'<section class="sp-section single">{body}</section>')

    # Otherwise, group by section/column indexes from each webpart's data.layout
    # Note: shape can vary; we’ll read 'zoneIndex' (section) & 'columnIndex' (column).
    out_sections: List[str] = []
    for s_idx, section in enumerate(sections):
        cols = section.get("columns") or [{"width": 12, "columnIndex": 0}]
        col_htmls: List[str] = []
        for c in cols:
            c_idx = c.get("columnIndex", 0)
            col_parts = []
            for wp in webparts:
                l = (wp.get("data") or {}).get("layout") or {}
                if l.get("zoneIndex", s_idx) == s_idx and l.get("columnIndex", 0) == c_idx:
                    col_parts.append(wp)
            inner = "\n".join(render_webpart(wp) for wp in col_parts) or "<!-- empty -->"
            col_htmls.append(f'<div class="sp-column sp-col-{c.get("width", 12)}">{inner}</div>')
        out_sections.append(f'<section class="sp-section">{"".join(col_htmls)}</section>')

    return base_html(title, "\n".join(out_sections))

def base_html(title: str, body_inner: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{escape(title)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;
         margin:0 auto;max-width:1200px;padding:16px;line-height:1.5}}
    h1{{margin:0 0 16px 0;font-size:1.6rem}}
    .sp-section{{display:flex;gap:16px;margin:0 0 24px 0}}
    .sp-section.single{{display:block}}
    .sp-column{{flex:1;min-width:0}}
    .sp-webpart{{border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:12px}}
    .sp-text p{{margin:0 0 8px 0}}
    details summary{{cursor:pointer}}
    pre{{white-space:pre-wrap;word-break:break-word}}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {body_inner}
</body>
</html>"""

# ---------------------------
# Optional: CanvasContent1 fallback (SharePoint REST)
# ---------------------------
def get_canvas_content1(
    token: str,
    host: str,               # e.g., "contoso.sharepoint.com"
    item_id: int,            # list item ID in "Site Pages" library
    site_relative_url: str = ""  # e.g., "sites/Marketing" (no leading slash)
) -> Dict[str, Any]:
    """
    Reads the Site Pages list item and returns CanvasContent1 + LayoutWebpartsContent.
    Note: This uses SharePoint REST (not Graph). Your token must be accepted by SP (same AAD app).
    """
    # Build REST url
    base = SP_REST_BASE_FMT.format(host=host)
    site_prefix = f"/{site_relative_url}" if site_relative_url else ""
    url = f"{base}{site_prefix}/web/lists/GetByTitle('Site Pages')/items({item_id})"
    params = {"$select": "Title,CanvasContent1,LayoutWebpartsContent,FileRef"}
    r = requests.get(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json;odata=nometadata"
    }, params=params, timeout=60)
    if not r.ok:
        raise RuntimeError(f"SharePoint REST failed: {r.status_code} {r.text}")
    return r.json()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    """
    Usage:
      export AZURE_TENANT_ID=...
      export AZURE_CLIENT_ID=...
      export AZURE_CLIENT_SECRET=...          # or set USE_DEVICE_CODE=1 for user sign-in
      python sp_page_reconstruct.py
    """
    import argparse
    parser = argparse.ArgumentParser(description="Fetch & reconstruct a modern SharePoint page")
    parser.add_argument("--site-id", required=True, help="Graph site ID (e.g., contoso.sharepoint.com,123,abc)")
    parser.add_argument("--page-id", required=True, help="Page ID (GUID-like) from Graph Pages API")
    parser.add_argument("--out", default="page.html", help="Output HTML file")
    parser.add_argument("--debug", action="store_true", help="Print raw JSON for page and parts")
    args = parser.parse_args()

    token = get_access_token()
    page = get_page(token, args.site_id, args.page_id)
    parts = get_page_webparts(token, args.site_id, args.page_id)

    if args.debug:
        print("=== PAGE ===")
        print(json.dumps(page, indent=2))
        print("=== WEBPARTS ===")
        print(json.dumps(parts, indent=2))

    html_out = reconstruct_html(page, parts)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"Wrote {args.out}")
