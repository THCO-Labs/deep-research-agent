import pytest
import json

from deep_research.scraper import ScrapeQualityError, extract_markdown_candidates, html_to_markdown
from deep_research.scraper import _meta_refresh_url


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


def test_html_to_markdown_rejects_access_denied_page() -> None:
    html = """
    <html>
      <head><title>Access denied</title></head>
      <body>
        <h1>403 Forbidden</h1>
        <p>Access denied. Your request was blocked by a security service.</p>
      </body>
    </html>
    """

    with pytest.raises(ScrapeQualityError):
        html_to_markdown(html, title="Access denied")


def test_html_to_markdown_rejects_structural_captcha_page() -> None:
    html = """
    <html>
      <head><title>Security check</title></head>
      <body>
        <form id="challenge-form">
          <iframe src="https://captcha.example/challenge"></iframe>
          <input name="captcha-token" value="">
        </form>
      </body>
    </html>
    """

    with pytest.raises(ScrapeQualityError):
        html_to_markdown(html, title="Security check")


def test_html_to_markdown_prefers_article_over_navigation() -> None:
    nav_links = "".join(f"<a href='/nav-{i}'>Navigation item {i}</a>" for i in range(100))
    html = f"""
    <html>
      <body>
        <header>{nav_links}</header>
        <main>
          <article>
            <h1>What are urban heat islands?</h1>
            <p>Urban heat islands occur when built environments retain heat and raise local temperatures.</p>
            <p>They are shaped by pavement, dense buildings, reduced vegetation, waste heat, and limited airflow.</p>
            <p>Public health teams study these patterns because heat exposure can increase illness and mortality risk.</p>
          </article>
        </main>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="What are urban heat islands?")

    assert "Urban heat islands occur" in markdown
    assert "Navigation item" not in markdown


def test_html_to_markdown_selects_wikipedia_article_body() -> None:
    html = """
    <html>
      <body>
        <nav>Main page Contents Current events Random article About Wikipedia</nav>
        <div id="bodyContent">
          <div class="mw-parser-output">
            <h1>Urban heat island</h1>
            <p>An urban heat island is an urban area that is significantly warmer than surrounding rural areas.</p>
            <p>The temperature difference is influenced by land cover, building materials, vegetation, and human activity.</p>
            <p>Mitigation strategies include tree canopy expansion, cool roofs, reflective pavement, and heat emergency planning.</p>
          </div>
        </div>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="Urban heat island")

    assert "An urban heat island" in markdown
    assert "Main page Contents" not in markdown


def test_html_to_markdown_skips_children_of_removed_noise_nodes() -> None:
    html = """
    <html><body>
      <main>
        <div class="newsletter">
          <p>This child should not crash after its parent is removed.</p>
        </div>
        <article>
          <h1>Urban heat islands</h1>
          <p>Urban heat islands raise neighborhood temperatures through heat-absorbing surfaces and reduced vegetation.</p>
          <p>The process can increase heat stress, worsen air quality, and strain public health services.</p>
          <p>Researchers measure surface temperature, air temperature, tree canopy, impervious cover, and population vulnerability.</p>
          <p>Cities often combine cooling centers, shade, reflective roofs, and emergency alerts to reduce risk.</p>
          <p>The resulting plan can prioritize neighborhoods where exposure and health vulnerability overlap.</p>
        </article>
      </main>
    </body></html>
    """

    markdown = html_to_markdown(html, title="Urban heat islands")

    assert "Urban heat islands raise" in markdown
    assert "This child should not crash" not in markdown


def test_html_to_markdown_extracts_article_body_from_json_ld() -> None:
    article_body = (
        "Urban heat islands are local temperature increases caused by heat-retaining "
        "surfaces, dense development, sparse vegetation, and waste heat. Public health "
        "departments study them because extreme heat can increase emergency visits, "
        "cardiovascular strain, respiratory risk, and mortality. Mitigation requires "
        "tree canopy, cool roofs, reflective materials, shaded transit stops, cooling "
        "centers, and targeted outreach for vulnerable residents."
    )
    html = f"""
    <html>
      <head>
        <script type="application/ld+json">
          {json.dumps({"@type": "TechArticle", "headline": "Urban heat island guide", "articleBody": article_body})}
        </script>
      </head>
      <body>
        <nav>Subscribe Login Search</nav>
        <div id="app"></div>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="Urban heat island guide")

    assert "Urban heat island guide" in markdown
    assert "Urban heat islands are local temperature increases" in markdown
    assert "Subscribe Login Search" not in markdown


def test_html_to_markdown_extracts_article_content_from_application_json() -> None:
    html_fragment = (
        "<p>Mediterranean-style eating patterns emphasize vegetables, fruits, legumes, "
        "whole grains, nuts, olive oil, and moderate fish intake.</p>"
        "<p>For adults with hypertension, studies often evaluate systolic pressure, "
        "diastolic pressure, sodium intake, potassium intake, and cardiovascular risk.</p>"
        "<p>Practical nutrition guidance combines diet quality, medication adherence, "
        "monitoring, physical activity, and clinician follow-up.</p>"
        "<p>Limits include cost, food access, cultural fit, allergies, and cases where "
        "dietary change alone is not enough.</p>"
    )
    payload = {"props": {"pageProps": {"article": {"headline": "Embedded article", "content": html_fragment}}}}
    html = f"""
    <html>
      <body>
        <script type="application/json">{json.dumps(payload)}</script>
        <main><p>Loading...</p></main>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="Embedded article")

    assert "Mediterranean-style eating patterns" in markdown
    assert "adults with hypertension" in markdown
    assert "Loading" not in markdown


def test_html_to_markdown_extracts_long_text_from_arbitrary_json_payload() -> None:
    embedded_text = (
        "Flood resilience planning combines hydrology, land use, drainage capacity, early warnings, "
        "insurance exposure, emergency shelters, and social vulnerability mapping. Local teams use "
        "this evidence to prioritize investments across neighborhoods, protect critical facilities, "
        "coordinate evacuation routes, and decide where green infrastructure can reduce runoff. "
        "The strongest plans compare historical flood losses, current infrastructure condition, "
        "climate projections, maintenance budgets, and community needs before recommending action."
    )
    payload = {"data": {"blocks": [{"kind": "unknown", "payload": {"bodyCopy": embedded_text}}]}}
    html = f"""
    <html>
      <body>
        <script type="application/json">{json.dumps(payload)}</script>
        <main><p>Loading application shell.</p></main>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="Flood resilience planning")

    assert "Flood resilience planning combines" in markdown
    assert "Loading application shell" not in markdown


def test_html_to_markdown_scores_nonstandard_content_container() -> None:
    html = """
    <html>
      <body>
        <div class="top-links">
          <a href="/one">Menu one</a><a href="/two">Menu two</a><a href="/three">Menu three</a>
        </div>
        <div data-layout="story">
          <h1>Community solar finance</h1>
          <p>Community solar finance depends on subscriber acquisition, interconnection timelines, tax credit monetization, and project debt terms.</p>
          <p>Developers compare household savings, anchor tenant contracts, grid hosting capacity, policy incentives, and operating expenses.</p>
          <p>Public agencies evaluate whether program rules protect low-income subscribers, reduce cancellation risk, and keep benefits local.</p>
          <p>Strong analysis also tracks permitting delays, utility billing rules, credit support, ownership structures, and long-term maintenance.</p>
        </div>
      </body>
    </html>
    """

    markdown = html_to_markdown(html, title="Community solar finance")

    assert "Community solar finance depends" in markdown
    assert "Menu one" not in markdown


def test_extract_markdown_candidates_returns_ranked_fallbacks() -> None:
    html = """
    <html>
      <body>
        <nav>Search Login Subscribe Newsletter</nav>
        <main>
          <article>
            <h1>Heat resilience planning</h1>
            <p>Urban heat resilience planning combines temperature mapping, tree canopy analysis, emergency alerts, and cooling centers.</p>
            <p>Public health agencies use these plans to prioritize neighborhoods with older adults, outdoor workers, chronic illness, and limited access to air conditioning.</p>
            <p>Strong plans compare heat exposure, social vulnerability, housing quality, medical risk, transportation access, and local response capacity.</p>
            <p>Evidence from city programs helps teams choose shade investments, cool roofs, reflective pavement, hydration outreach, and overnight cooling options.</p>
          </article>
        </main>
      </body>
    </html>
    """

    candidates = extract_markdown_candidates(html, title="Heat resilience planning")

    assert candidates
    assert candidates[0].score >= candidates[-1].score
    assert any("Urban heat resilience planning" in candidate.markdown for candidate in candidates)
    assert candidates[0].method in {
        "structured_json",
        "trafilatura_balanced",
        "trafilatura_precision",
        "trafilatura_recall",
        "readability",
        "newspaper4k",
        "goose3",
        "justext",
        "selectolax",
        "inscriptis",
        "beautifulsoup",
        "lxml_text",
    }


def test_meta_refresh_url_resolves_relative_target() -> None:
    html = """
    <html>
      <head><meta http-equiv="refresh" content="0; URL='/articles/full-report'"></head>
      <body>Moved</body>
    </html>
    """

    assert _meta_refresh_url(html, "https://example.org/start") == "https://example.org/articles/full-report"
