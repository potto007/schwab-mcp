# Security Policy

`schwab-mcp` lets an AI model read Schwab brokerage accounts and, when write
tools are enabled, place orders after a human approves them. Please report
security problems privately.

This repository (`potto007/schwab-mcp`) is a fork of
[`jkoelker/schwab-mcp`](https://github.com/jkoelker/schwab-mcp), maintained by
one person. This policy covers the fork only.

## Supported versions

| Version | Supported |
| --- | --- |
| The `main` branch of this repository | Yes |
| Container images at `ghcr.io/potto007/schwab-mcp` built from `main` or from the newest release tag | Yes |
| Older commits, older image tags, and other branches | No. Update to `main` first. |
| `jkoelker/schwab-mcp` and images at `ghcr.io/jkoelker/schwab-mcp` | Not covered here. See "Issues that also affect upstream". |

Fixes land on `main`. There are no backports.

## Reporting a vulnerability

Use GitHub private vulnerability reporting:
<https://github.com/potto007/schwab-mcp/security/advisories/new>. You can also
open the repository's **Security** tab and choose **Report a vulnerability**.

Do not open a public issue, pull request, or discussion for a suspected
vulnerability, and do not describe one in a comment on an existing issue or
pull request. If you are not sure whether something is a vulnerability, report
it privately anyway.

Include:

- the commit SHA or image digest you tested;
- how the server was run: stdio or HTTP, which approval backend, and the
  flags that matter;
- steps to reproduce, ideally as a test against the fake clients in `tests/`;
- what an attacker gains, and what they need to already have.

Redact account numbers, account hashes, tokens, client secrets, bot tokens and
phone numbers. A report never needs real ones.

## Testing rules

- Reproduce with the test suite's fake clients or with mocks of your own. No
  finding needs a live order to prove it.
- Never place, replace or cancel a live order to demonstrate a finding, even
  in your own account.
- Never use credentials, tokens, brokerage accounts, Discord servers or Signal
  numbers that are not yours.
- Do not scan, fuzz or load-test Schwab, Discord, Signal or GitHub.
- Do not try to social-engineer the maintainer or an approver.
- If you come across someone else's data, stop, do not keep it, and say so in
  the report.

## Scope

In scope:

- The approval flow: anything that makes the order placed, replaced or
  cancelled differ from what the reviewer was shown, or that performs a write
  without an approval.
- The approval backends as implemented here: who can approve, and prompts
  that can be spoofed, truncated or replayed.
- Validation of tool inputs before they reach an approval prompt or a Schwab
  request.
- The HTTP transport: exposure beyond what the documentation states, DNS
  rebinding, and cross-origin requests.
- Storage and handling of tokens and credentials, including leaks into logs,
  error messages or approval prompts.
- The container image built from this repository: its contents, user and
  defaults.
- Packaging and dependency pins, including a pinned dependency version with a
  known vulnerability that this project can reach.
- Build and automation configuration in this repository.
- Issues in code inherited from upstream, when they reproduce on this fork's
  `main`.

Out of scope:

- Vulnerabilities in Schwab's API, Discord, Signal, GitHub or third-party
  libraries themselves. Report those to their maintainers.
- The documented effect of options that are documented as unsafe:
  `--jesus-take-the-wheel` (no approval), `--http-allow-remote` and
  `--signal-allow-remote`. Getting their effect without passing the flag is in
  scope.
- An AI model proposing a bad or unwanted trade that the approval prompt shows
  accurately. The prompt is the control, so a report has to show the prompt
  being wrong, missing or bypassed.
- Attacks that need an already compromised host, operating-system account,
  approver account or MCP client.
- Scanner output that lists base-image or OS-package CVEs with no demonstrated
  effect on `schwab-mcp`.

## What to expect

This fork has one maintainer working in spare time. These are aims, not
guarantees:

- acknowledgement within 7 days;
- an assessment, with severity and a plan, within 21 days of the report;
- a fix as soon as the severity warrants. Approval bypass and order integrity
  come first.

Updates are posted on the private advisory. If you hear nothing for 14 days,
comment on the advisory rather than going public. There is no bug bounty.
Reporters are credited in the published advisory unless they ask not to be.

## Disclosure

Please keep a report confidential until a fix is on `main` and the advisory is
published, or until 90 days after the report, whichever comes first. If more
time is needed, we agree on a new date on the advisory.

A fixed issue is published as a GitHub Security Advisory on this repository,
naming the affected commits and the commit that fixes it.

## Issues that also affect upstream

Most of this code comes from upstream, so a problem found here may exist there
too.

- If it reproduces on this fork's `main`, report it here, and say whether you
  also tested upstream.
- The report is tracked here as a private advisory.
- The maintainer forwards it to the upstream maintainer and coordinates the
  timing of the fix and the disclosure with them. Nothing that would expose an
  unfixed upstream issue is published here before the agreed date.
- You are welcome to report to upstream yourself. This fork cannot accept
  reports on upstream's behalf and does not control upstream's response.
  Please do not open a public upstream issue for something you reported here
  while it is unfixed.
- Problems in code that exists only in this fork are handled here alone.

## Container images

- Images at `ghcr.io/potto007/schwab-mcp` are built by this repository's CI.
  The `main` tag follows the `main` branch. `latest` and version tags follow
  release tags.
- Images at `ghcr.io/jkoelker/schwab-mcp` are built by upstream and do not
  contain this fork's changes.
- Pin an image by digest, and give that digest in a report.
- An image is rebuilt only when `main` or a release tag changes. Its base
  image and OS packages are as of that build. For current OS packages, pull a
  newer image or build from the `Containerfile`.
