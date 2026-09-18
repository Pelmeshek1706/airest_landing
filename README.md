# AIREST landing site

Standalone React + TypeScript + Vite landing page, prepared on the `landing_new` branch. Includes the animated assessment monitors, analytics wrap-up, face animation, and approved hybrid contact form. The existing `main` branch is unchanged.

## Start locally

Install Node.js 24 (or run `nvm use` if you use nvm), then:

```sh
git clone --branch landing_new --single-branch https://github.com/Pelmeshek1706/airest_landing.git
cd airest_landing
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Stop with Ctrl+C.

## Build and host

```sh
npm run build
npm run preview
```

Preview the production build at http://127.0.0.1:4173. `npm run build` checks TypeScript and creates `dist/`.

Upload **the contents of `dist/`**, including its `assets/` and `fonts/` folders, to your static host's public directory. Serve it over HTTP(S); opening `index.html` directly as a file will not run the application correctly. No Node.js server or database is required in production.

For a Git-connected static host, select branch `landing_new`, use `npm ci && npm run build` as the build command, `dist` as the output directory, and Node.js 24 as the build runtime. Set any environment variables before building. Deploying this branch is a separate hosting action; pushing it does not change `main`.

Asset paths are relative, so the build supports a domain root or a subdirectory. Use the trailing slash when hosting in a subdirectory. The existing domain `www.airest.health` is preserved in `public/CNAME`, which is copied into the build. Edit or remove that file if deploying to a different GitHub Pages domain. For GitHub Pages, publish the **built `dist/` contents** through a Pages deployment workflow or a dedicated output branch; the source branch itself is not a ready-to-serve website.

Enable Brotli or gzip on the host. Cache versioned JavaScript/CSS files for a long time and let HTML revalidate. Public artwork filenames are not versioned; avoid immutable caching for those files. Artwork and video account for most of the roughly 15 MB uncompressed build; lazy loading means it is not all requested at once.

## Contact form

By default, the form opens an email draft addressed to **airest.operating@gmail.com**. Visitors must send it from their email app; the site does not send email itself.

To configure it, copy `.env.example` to `.env.local`, change values, and rebuild:

```sh
cp .env.example .env.local
```

- `VITE_CONTACT_EMAIL`: recipient for the email draft.
- `VITE_CONTACT_ENDPOINT`: optional HTTPS endpoint that receives a JSON POST containing `name`, `organisation`, `email`, and `interests` (an array of selected labels). When set, it takes precedence over the email draft. Any successful HTTP status is treated as acceptance. The receiving service must validate submissions, handle delivery, and permit the site's origin through CORS when necessary. No backend is included.

All `VITE_` variables become public browser code. Never place passwords or private API keys in them.

## Included files

- `src/`: landing page, contact form, monitor demo, animation, and styles.
- `public/assets/`: artwork and media used by the landing page.
- `public/fonts/`: self-hosted fonts and their license notices.
- Package lockfile and build configuration for reproducible installation.

Design comparison routes, source design exports, clinical application files, local environment settings, dependencies, and generated builds are excluded. The old site's files remain accessible in Git history and on `main`. The monitor assessment and analytics are illustrative demo content.
