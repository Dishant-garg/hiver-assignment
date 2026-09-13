# Intent taxonomy — SpotifyCares

Nine intents plus `other`, derived from `data/processed/SpotifyCares/corpus.csv`
(8,000 pairs; **4,952 are first-turn**, `turn_index == 0`). The holdout was not read:
it feeds the evaluation set.

Shares below are the **weak keyword labeller** (`intents.weak_label`) over those 4,952
first-turn messages — an estimate of prevalence, not ground truth. The labeller exists to
stratify sampling and to give the baseline something to beat; the LLM classifier reads the
definitions and examples, never the keywords. A vote is one *non-overlapping matched span*,
so a message saying "double charged" scores `billing_and_refund` once, not twice.

## The taxonomy

| intent | definition | auto-handle | typical brand resolution (verbatim reply from the corpus) | est. share |
|---|---|---|---|---|
| `account_access` | Login/password reset fails, hacked, deletion. | **No** | "Hey, help's here! Can you DM us your account's username or email address? We'll take a look under the hood /RS" | 6.8% |
| `billing_and_refund` | Wrong charge, failed payment, refund; plan change → `subscription_plan`. | **No** | "Hey there! Can you DM us your account's username and email address? We'll take a look backstage /SY" | 7.5% |
| `subscription_plan` | Premium/Family/Student/trial plan or cancel; charge → `billing_and_refund`. | **No** | "Hey! Can you let us know which university you're currently attending? More info about the student discount here: /NQ" | 10.7% |
| `playback_issue` | Saved content won't play or skips; crashes → `app_bug`. | Yes | "Hey Dean! Best thing to try here is a reinstall. Just follow the steps at Let us know how it goes /MX" | 7.2% |
| `app_bug` | Client crashes, freezes or won't open; audio → `playback_issue`. | Yes | "Hey Mark, help's here! Can you let us know your exact OS X and Spotify versions? We'll see what we can suggest /JP" | 7.4% |
| `library_playlist_issue` | Own saved songs/playlists/downloads gone; catalogue → `content_availability`. | Yes | "Hey, sorry to hear that! Check out the steps under “Downloads unexpectedly removed” at They should help /RH" | 10.7% |
| `content_availability` | Song/album/artist/country never on Spotify or pulled; own saves → `library_playlist_issue`. | Yes | "Hey Anders! Sometimes content gets temporarily removed because of licensing changes. Hopefully we'll have it available again soon /JX" | 11.4% |
| `feature_request` | Wants a feature that doesn't exist or is retired. | Yes | "Hi Shawn. Check out the official idea and add your vote to let our devs know it's something you'd like to see: /JU" | 3.6% |
| `how_to_question` | Asks how to use a feature that works fine. | Yes | "Hi there! Can you head over to Settings > Language > Choose Language and select the one that works for you? Keep us posted /NY" | 2.5% |
| `other` | Praise, rants, pitches, or too vague to act on. | **No** | "Hey, help's here! Can you tell us what's happening exactly and on what device? We'll see what we can suggest /JM" | 32.1% |

Distribution check (Step 4 gate: `other` ≤ 40%, every other intent ≥ 2%) — **passes**:
`other` 32.1%, minimum non-other 2.5%.

`how_to_question` is the one intent that does **not** clear the brief's stricter ≥ 3% rule.
It did (3.8%) before the weak labeller was fixed to count non-overlapping spans; roughly a
third of its old share was the double-count artefact of "how do i" *and* "how do" both firing
on the same four words. The honest number is 2.5%, and the intent is kept because it has a
distinct resolution (link the help-centre article / name the setting) that no other intent
covers — see the reply quoted above.

### Boundary rules for the three close pairs

The definitions carry an explicit `-> sibling` pointer, because these are the boundaries a
labeller — human or LLM — actually gets wrong. They are the deciding question in each case:

- **`billing_and_refund` vs `subscription_plan`** — is the complaint about *money that moved*
  (wrong, double, unexpected charge; failed payment; refund) or about *which plan the account
  is on* (student verification, family invite, upgrade, cancel)? "You charged me twice" is
  billing; "I can't add my son to the family plan" is subscription.
- **`playback_issue` vs `app_bug`** — the content exists and is saved, but won't come out of
  the speaker → playback. The client itself crashes, freezes, won't open or won't update →
  app_bug. "It skips every ten seconds" is playback; "it crashes when I hit share" is app_bug.
- **`content_availability` vs `library_playlist_issue`** — whose content is missing? Never on
  Spotify, or pulled from the catalogue → content_availability. The customer's *own* saved
  songs, playlists or downloads vanished → library_playlist_issue. "Why isn't this album on
  Spotify" vs "all my saved songs are gone".

## Why each auto-handle flag is what it is

The flags are not a guess. For each weak-labelled intent I measured what the brand's own
first reply actually does: `dm_deflect%` = the reply hands the customer off out of the
thread (DM, "reach out", "contact us", a bare link); `resolution_step%` = the reply contains
a concrete step (reinstall, restart, log out, clear cache, settings, "steps at…"). Both use
the same regexes as the brand survey. Reproduce with `scripts/intent_deflection.py`.

| intent | n | dm_deflect% | resolution_step% | auto-handle |
|---|---|---|---|---|
| `account_access` | 337 | **79.5** | 21.4 | No |
| `billing_and_refund` | 373 | **88.2** | 11.0 | No |
| `subscription_plan` | 532 | **74.6** | 12.6 | No |
| `playback_issue` | 357 | 17.9 | 15.7 | Yes |
| `app_bug` | 367 | 21.5 | 14.7 | Yes |
| `library_playlist_issue` | 529 | 23.1 | 20.2 | Yes |
| `content_availability` | 565 | 14.3 | 12.4 | Yes |
| `feature_request` | 179 | 8.9 | 20.1 | Yes |
| `how_to_question` | 123 | **39.0** | 19.5 | Yes (weakest call) |
| `other` | 1590 | 28.9 | 12.5 | No |

The split is clean: the three intents marked `auto_handle_allowed=False` are the only ones
where the brand hands the customer off in roughly three replies out of four, and they are
exactly the three that require **account access, money movement, or a policy exception** —
you cannot answer "why was I charged twice" or "my account was hacked" in public without
looking at the account. The six `True` intents are answered in-thread 61–91% of the time
with a standard fix (reinstall, ask for OS + app version, link the "Downloads unexpectedly
removed" article, explain licensing, log the idea).

Per-intent justification:

- **`account_access` — No.** Needs account access by definition. Also the highest-harm
  failure mode: a confident wrong answer to a hacked-account report is worse than silence.
- **`billing_and_refund` — No.** Money movement, and the most deflected intent in the whole
  corpus (88.2%). The agent cannot see a charge, issue a refund, or promise one.
- **`subscription_plan` — No.** Student verification, family invites and plan switches are
  account operations, and "can you make an exception for me" is a policy exception. A few
  members of this class are pure how-to ("can we pay a family plan upfront?"), which is the
  cost of the boundary — accepted, because getting a plan change wrong is expensive.
- **`playback_issue` — Yes.** The brand's standard in-thread fix is real and repeatable:
  reinstall, check platform/device, check offline mode. 17.9% deflected.
- **`app_bug` — Yes.** Same: ask for OS + Spotify version, suggest reinstall. No account
  data needed to give the first, correct step.
- **`library_playlist_issue` — Yes.** The published self-serve answers ("Downloads
  unexpectedly removed", the 3,333-offline-songs-per-device limit, the playlist recovery
  page) cover most of it, and it has the highest resolution-step share of any auto-handleable
  intent (20.2%). Caveat: genuinely lost libraries still get pulled into DM (23.1%), so the
  downstream confidence and retrieval thresholds are what catch the cases where no good
  precedent is retrieved.
- **`content_availability` — Yes.** Lowest deflection of any problem intent (14.3%). The
  answer is a policy statement, not an account action: licensing varies by region, here's
  the info page, we'll pass the request on.
- **`feature_request` — Yes.** Lowest deflection overall (8.9%), with a resolution-step
  share near the top of the table (20.1%). The standard resolution is "noted, vote on the idea here" —
  safe to automate, as long as the agent never promises a ship date.
- **`how_to_question` — Yes, with the least confidence.** 39.0% of these still went to DM,
  because a fair number of "how do I…" messages are account operations in disguise ("how do
  I switch my payment to my Starbucks number"). Kept `True` because the majority really are
  answerable from the help centre; the confidence gate is what protects the rest. If the
  eval shows this intent driving wrong auto-handles, flipping it to `False` is a one-line
  change.
- **`other` — No, always.** By construction it is the bucket for messages the agent did not
  understand; auto-handling an un-understood message is the exact failure this gate exists
  to prevent.

## How the taxonomy was derived

**Step 1 — clustering.** `scripts/explore_intents.py 12` fits TF-IDF (1–2 grams, `min_df=5`,
English stop words, sublinear tf) over the first-turn messages and KMeans with `k=12`,
`random_state=42`, printing the top 12 terms and 15 sampled messages per cluster. Then 200
random first-turn messages were read by hand (`random_state=1` and `random_state=7` — two
seeds, so the read is not one lucky draw).

**Clusters as they came out** (n = first-turn messages, from the pre-unescape build):

| cluster | n | top terms | what it actually was |
|---|---|---|---|
| 0 | 1543 | hey, playlists, like, albums, guys, fix | grab-bag: praise, jokes, job pitches, petitions, business dev |
| 1 | 204 | student, discount, charged, hulu, 99 | student plan + student billing |
| 2 | 401 | premium, account, family, free, charged | Premium/Family plan state |
| 3 | 418 | songs, playlist, downloaded, phone | the customer's own library/playlists |
| 4 | 257 | song, play, playing, playlist, available | playback + "this song won't play" |
| 5 | 173 | album, new album, listen, available | catalogue gaps and new releases |
| 6 | 181 | need, need help, help, asap | **no topic at all** — pure urgency |
| 7 | 344 | app, apple, watch, android, desktop | app bugs *and* Apple Watch feature requests |
| 8 | 500 | account, family, email, password, hacked | login/hacked + family plan |
| 9 | 227 | free, 30, ad, trial, minutes | free-tier ad complaints + trials |
| 10 | 452 | spotify, available, connect, phone | catalogue + country launches |
| 11 | 301 | music, spotify, listen, apple music | switching threats, library, recommendations |

**Merges made (clusters → intents).** KMeans clustered on *vocabulary*; the taxonomy had to
cut on *what is broken and who can fix it*, so the mapping is not one-to-one.

1. **Clusters 3/4/5/10/11 all say "song / album / playlist"** and had to be split three
   ways by *whose* content is missing: content that was never on Spotify or was pulled →
   `content_availability`; content the customer saved that vanished → `library_playlist_issue`;
   content that exists and is saved but won't come out of the speaker → `playback_issue`.
   These three have visibly different brand replies (licensing explainer vs. the
   "Downloads unexpectedly removed" article vs. reinstall/platform question), which is what
   justifies keeping them apart.
2. **Clusters 1/2/8 were one "account" blob** and were split three ways by what the reply
   asks for: `account_access` ("DM your email, we'll look backstage"), `billing_and_refund`
   ("DM your username *and* email"), `subscription_plan` ("which university are you at?").
   The three-way split of the deflection numbers above confirms it.
3. **Cluster 7 was split in half**: broken client → `app_bug`; "please build an Apple Watch
   app" → `feature_request`. They share the word "app" and nothing else; the replies are
   "which OS/version?" vs. "vote on the idea".
4. **Cluster 6 ("I need help ASAP", n=181) was not made an intent.** It has no topic — it
   is the urgency register, not a subject. Those messages land in `other` (or in a real
   intent when the customer adds detail), and `other` is never auto-handled, which is the
   right outcome for a message that says nothing.
5. **Cluster 0 (n=1543) is most of the 32.1% `other`.** Reading it confirmed it is not a
   missed intent: it is praise, jokes, recruitment ("It smell like y'all hiring?"),
   partnership pitches, and content-curation opinions. Nothing there has a support
   resolution, so a 32% `other` rate is a property of Twitter support traffic, not a gap.

**Intents considered and rejected:**

- **`ads_complaint`** (cluster 9: "30 min ad free and an ad played after 18", "these ads are
  too loud"). Real volume, but no distinct resolution — the brand either explains the free
  tier, points at Premium, or says nothing actionable. It is either a rant (`other`), a
  playback bug (`playback_issue`), or a Premium upsell (`subscription_plan`). Splitting it
  out would have created an intent whose only correct reply is "upgrade", which is a sales
  answer, not a support one.
- **`device_integration`** (Sonos, Chromecast, Ford Sync, Roku, car). Tempting, and the
  clusters show it, but the resolution is *identical* to `app_bug` (ask OS/app version,
  reinstall) or `feature_request` (no app exists yet). An intent that shares its resolution
  with another intent earns nothing and costs classifier confusion.
- **`artist_or_royalty_issue`** (artists whose track is filed under the wrong profile,
  earnings reports). It *does* have a distinct resolution ("send us the artist page links" /
  Spotify for Artists), but it is well under 3% of first-turn traffic and it is a different
  product surface. Folded into `other`, which escalates — the safe outcome.
- **`country_launch`** ("launch in India/Kenya/Russia"). Distinct reply ("add your email to
  stay in the loop") but under 3% alone, and it is the same customer question as "why isn't
  this album available here". Merged into `content_availability`.
- **`praise_or_thanks`**. Frequent, but a compliment needs no resolution, so it would be an
  intent whose entire job is to not act. That is what `other` already does.
- **`non_english`**. Only 0.8% of SpotifyCares customer messages look non-English (measured
  in Task 2), and language is a property of a message, not an intent.

**Why 9 + `other` and not fewer.** Collapsing `billing_and_refund` into `subscription_plan`
was tried on paper and rejected: a refund request and a family-invite failure escalate for
different reasons and would read very differently in a reply, and both are large (7.5% and
10.7%). Collapsing `app_bug` into `playback_issue` was likewise rejected — the brand's
diagnostic question differs (device/OS vs. reinstall), and merging would produce one 14.6%
intent that the retriever could not serve precedents for cleanly.

## Known weaknesses

- The weak labeller is keyword voting with ties broken by list order, so the three
  escalate-always intents are listed **first** in `INTENTS` deliberately: a message that hits
  both "refund" and "how do i" is labelled `billing_and_refund`, not `how_to_question`.
- Votes count non-overlapping spans rather than keywords, so nested phrases ("double charged"
  + "charged", "how do i" + "how do") score once. This *lowered* several shares versus the
  first pass — most visibly `how_to_question`, 3.8% → 2.5% — which means the first pass was
  overstating intents that happened to have long phrases in their keyword lists.
- `other` at 32.1% is partly real (cluster 0) and partly labeller recall — messages like
  "I got kicked out of my premium Spotify and it won't let me back in" are clearly
  `account_access` but contain none of its keywords. The LLM classifier is expected to beat
  the labeller here; that gap is the point of the baseline comparison.
- `how_to_question` is the least separable intent (39.0% of its brand replies still deflect)
  and the most likely candidate to be flipped to `auto_handle_allowed=False` after eval.
- `definitions_block()` ships only the *first* example per intent (1,148 chars, ~255 tokens);
  the second, longer example on each `Intent` is kept for docs and few-shot use and is not
  paid for on every classify call.

## Reproducing

```bash
.venv/bin/python scripts/intent_deflection.py     # the dm_deflect% / resolution_step% table
.venv/bin/python scripts/explore_intents.py 12 > /tmp/intent_explore.txt
.venv/bin/python -c "import pandas as pd; from support_agent import config, intents; \
d=pd.read_csv(config.brand_dir()/'corpus.csv'); d=d[d.turn_index==0]; \
print(d.customer_text.map(intents.weak_label).value_counts(normalize=True).round(3))"
```
