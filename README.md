# Vortex RFCs

This repo is used to store and discuss design decisions, features and significant changes in Vortex, either in specific APIs, 3rd-party integrations or the conceptual model.

## What needs an RFC?

Vortex is a rapidly changing open source project that provides a file format, a compute toolkit, and a set of Rust libraries implementing a range of state-of-the-art compression codecs.

Most development decisions in Vortex happen through the Discussions functionality. Simply open a new discussion describing the kind of change you'd like to make to the project, and if the maintainers like it, open a PR.

This is a great process for bug fixes, simple features and purely additive changes.

However, if you are making a significant change to the way Vortex functions, you may need to write an RFC first. Some example changes that would require an RFC:

* Making a risky change to the format, such as adding new required fields to file metadata
* Rearchitecting core components of Vortex, or wide-ranging refactors that might break language bindings
* Creating new libraries or SDKs that we expect others to adopt
* Making changes to subsystems that are likely to affect performance if not done thoughfully, such as the core IO traits

## Process

The general process is that changes should be opened as PRs, with the discussion happening in comments on top of it.

The PR should contain a markdown file with your proposal, in the `proposals/` directory. The file's name should start with a padded four digit number with the PRs number (`0007-my-proposal.md`), so that the proposals are sorted by submission time.

There's no set template but it should at the very least include the following details:

1. Proposal author(s).
1. Date of submission.
1. Informative name and a short description.

For changes that affect serialization, please take special care at explaining how compatibility is maintained and tested.

Once an agreement is achieved and concerns are addressed, the proposal will be merged into the repo.

## Building the Site

The RFCs are published as a static website. You'll need [Bun](https://bun.sh) installed.

### Install dependencies

```sh
bun install
```

### Development

Run the dev server with live reload:

```sh
bun run dev
```

Open http://localhost:3000 to view the site. Changes to files in `proposals/` or `styles.css` will automatically rebuild and refresh the browser.

### Production build

Generate the static site:

```sh
bun run build
```

The output is written to `dist/`. This folder can be deployed to any static hosting service (Cloudflare Pages, GitHub Pages, S3, Netlify, etc.).

### Clean

Remove the build output:

```sh
bun run clean
```
