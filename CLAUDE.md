# hiddenlayers.tech — Public Site Repo

This is the public GitHub Pages repository for [hiddenlayers.tech](https://hiddenlayers.tech), a personal site by Nate Cibik.

## Sibling Repo

The internal planning repo (`hiddenlayers-internal/`) contains roadmaps, research materials, task files, and draft content. In a multi-root workspace, reference materials there when making changes here. Never copy internal planning docs into this public repo.

## Stack

- **Jekyll 4.3.2** on **Ruby 3.1.2**, forked from `devlopr-jekyll` theme
- **Hosting:** GitHub Pages with CNAME → hiddenlayers.tech
- **Deploy:** GitHub Pages builds Jekyll natively. `DEPLOY_STRATEGY` is set to `none` (the Actions workflow is a no-op).
- **Markdown:** Kramdown with Rouge syntax highlighting and KaTeX math (`usemathjax: true` in post frontmatter)
- **Comments:** Hyvor Talk
- **Contact form:** Getform

## Key Directories

- `_posts/` — Blog posts (markdown with YAML frontmatter)
- `_authors/` — Author metadata (currently just `natecibik.md`)
- `_products/` — Shop items (hidden from nav, keep for future use)
- `_data/` — Site data files (authors.yml, galleries)
- `_includes/`, `_layouts/`, `_sass/` — Theme templates and styles
- `assets/img/posts/` — Blog post images
- `gallery/` — AI artwork gallery (keep, may expand later)

## Blog Post Format

```yaml
---
layout: post
title: "Post Title"
summary: "Short description"
author: natecibik
date: 'YYYY-MM-DD HH:MM:SS +0530'
category: CategoryName
thumbnail: /assets/img/posts/image.jpg
keywords: comma, separated, keywords
permalink: /blog/post-slug/
usemathjax: true
---
```

## Rules

- Do NOT delete infrastructure that may be useful later: shop, newsletter (mailchimp), gallery, comments, Docker configs.
- Do NOT modify the 3 original blog posts (`_posts/2023-09-*`) unless explicitly asked.
- Images for posts go in `assets/img/posts/`. Use descriptive filenames, not hashes.
- Test changes locally with `bundle exec jekyll serve --livereload` before committing.
- The site subtitle, author bio, and about page are being updated — check `_config.yml` and `_authors/natecibik.md` for current state.
