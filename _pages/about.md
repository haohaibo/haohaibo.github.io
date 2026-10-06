---
layout: about
title: about
permalink: /
# subtitle: Alibaba

profile:
  align: right
  image: haibo.png
  image_circular: false # crops the image to make it circular

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<!-- Homepage styles ported from sustcsonglin.github.io (_sass/_layout.scss, the .home-about section). They are inlined
     here because the style contract (test/style_contract.js) forbids owning _sass/ or _layouts/ in this repo. -->

<style>
  .intro-copy {
    max-width: 40rem;
    margin-bottom: 1.5rem;
    font-size: 1.08rem;
  }

  .intro-copy p {
    margin: 1.5rem 0 0;
  }

  .resource-grid {
    display: grid;
    gap: 0.75rem;
    margin: 1.15rem 0 2rem;
  }

  .resource-card {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.55rem 0.8rem;
    border: 1px solid var(--global-divider-color, rgba(0, 0, 0, 0.1));
    border-radius: 0.9rem;
    background: var(--global-card-bg-color, transparent);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    text-decoration: none;
    transition:
      border-color 180ms ease,
      box-shadow 180ms ease,
      transform 180ms ease;
  }

  .resource-card:hover {
    border-color: var(--global-theme-color, currentColor);
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.08);
    text-decoration: none;
    transform: translateY(-1px);
  }

  .resource-card > i {
    margin-top: 0.18rem;
    color: var(--global-theme-color, currentColor);
    font-size: 1.05rem;
  }

  .resource-card strong {
    display: block;
    margin-bottom: 0;
    line-height: 1.3;
  }

  .resource-card small {
    display: block;
    color: var(--global-text-color-light, inherit);
    font-size: 0.85rem;
    line-height: 1.3;
  }
</style>

<div class="intro-copy">
  <p><strong>Haibo (海波)</strong> is a software engineer at Alibaba, working on language model training performance. He worked as a performance architect in the NVIDIA Compute Architecture group from 2018 to 2025.</p>
  <p>He earned his bachelor's degree from
  <a href="https://www.xidian.edu.cn/">Xidian University</a> in 2015 and his master's degree from the <a href="http://www.ucas.ac.cn/">University of Chinese Academy of Sciences</a> in 2018, where he studied and worked at the
  <a href="http://www.ict.ac.cn/">Institute of Computing Technology, Chinese Academy of Sciences</a> from 2016 to 2018.</p>
</div>

---

<!-- Link cards in the style of sustcsonglin.github.io. Uncomment and fill in when there are projects or resources to feature.

<div class="resource-grid">
  <a class="resource-card" href="https://github.com/...">
    <i class="fa-brands fa-github fa-fw"></i>
    <span>
      <strong>Project name</strong>
      <small>one-line description</small>
    </span>
  </a>
</div>
-->
