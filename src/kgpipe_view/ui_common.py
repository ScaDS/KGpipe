from __future__ import annotations

import json

import streamlit.components.v1 as components


def render_mermaid(mermaid_text: str, height: int = 900) -> None:
    """Render Mermaid source in Streamlit using Mermaid JS."""
    mermaid_json = json.dumps(mermaid_text)
    html = f"""
    <style>
      .mermaid-wrapper {{
        background: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 8px;
        padding: 16px;
      }}
      .mermaid-wrapper .mermaid {{
        color: #111111;
      }}
    </style>
    <div class="mermaid-wrapper">
      <div class="mermaid"></div>
    </div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: false, securityLevel: "loose" }});
      const element = document.querySelector(".mermaid");
      element.textContent = {mermaid_json};
      await mermaid.run({{ nodes: [element] }});
    </script>
    """
    components.html(html, height=height, scrolling=True)
