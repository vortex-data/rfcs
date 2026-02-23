# Vortex RFCs

This repo is used to store and discuss design decisions, features and significant changes in Vortex, either in specific APIs, 3rd-party integrations or the conceptual model.

## Process

The general process is that changes should be opened as PRs, with the discussion happening in comments on top of it.

The PR should contain a markdown file with your proposal, in the `proposals/` directory. The file's name should start with a padded four digit number with the PRs number (`0007-my-proposal.md`), so that the proposals are sorted by submission time.

There's no set template but it should at the very least include the following details:

1. Proposal author(s).
1. Date of submission.
1. Informative name and a short description.

For changes that affect serialization, please take special care at explaining how compatibility is maintained and tested.

Once an agreement is achieved and concerns are addressed, the proposal will be merged into the repo.
