import pytest

from deep_research.scraper import ScrapeQualityError, html_to_markdown


def test_html_to_markdown_rejects_cloudflare_challenge() -> None:
    html = """
    <html>
      <head><title>Just a moment...</title></head>
      <body>
        <h1>Just a moment...</h1>
        <p>Performing security verification</p>
        <p>Enable JavaScript and cookies to continue</p>
      </body>
    </html>
    """

    with pytest.raises(ScrapeQualityError):
        html_to_markdown(html, title="Just a moment...")


def test_html_to_markdown_prefers_article_over_navigation() -> None:
    nav_links = "".join(f"<a href='/nav-{i}'>Navigation item {i}</a>" for i in range(100))
    html = f"""
    <html>
      <body>
        <header>{nav_links}</header>
        <main>
          <article>
            <h1>What is Fine-Tuning?</h1>
            <p>Fine-tuning is the process of adapting a pre-trained model to a specific downstream task.</p>
            <p>It usually uses a smaller task-specific dataset and updates either all model weights or a selected subset.</p>
            <p>This lets teams reuse general representations learned during pre-training while specializing the model.</p>
          </article>
        </main>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="What is Fine-Tuning?")

    assert "Fine-tuning is the process" in markdown
    assert "Navigation item" not in markdown


def test_html_to_markdown_selects_wikipedia_article_body() -> None:
    html = """
    <html>
      <body>
        <nav>Main page Contents Current events Random article About Wikipedia</nav>
        <div id="bodyContent">
          <div class="mw-parser-output">
            <h1>Fine-tuning (deep learning)</h1>
            <p>In deep learning, fine-tuning is the process of adapting a model trained for one task to perform a different, usually more specific, task.</p>
            <p>The additional training can be applied to the entire neural network or only to selected layers while other layers are frozen.</p>
            <p>Low-rank adaptation is a parameter-efficient variant that tunes smaller adapter weights while leaving the base model mostly unchanged.</p>
          </div>
        </div>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="Fine-tuning (deep learning)")

    assert "In deep learning, fine-tuning" in markdown
    assert "Main page Contents" not in markdown
