# Data

`gold_data.tsv` contains expert annotated data, while `ann_data.tsv` is a subset of that dataset annotated by 3 crowd annotators for the purposes of understanding agreement with our novel tagging framework for studying the intergroup bias. These datasets contain the following columns, with each row being an annotation for a comment:

- `split`: `train` or `test` used in modeling.
- `post_id`: The unique ID for the post (thread) this comment is from &mdash; the ID is assigned by the Reddit API, which we access using [pmaw](https://github.com/mattpodolak/pmaw).
- `comment_id`: Unique ID for the comment within a post.
- `parent_id`: `comment_id` for the parent comment if the current comment is a reply. The value is `t3_` + `post_id` if its a top-level comment (i.e. a reply to the thread itself).
- `tagged_comment`: The comment with referring expressions replaced by appropriate intergroup tags.
- `ref_expressions`: The corresponding referring expressions that were replaced with tags, in sentence word order.
- `ref_pos`: The positional spans of the referring expressions in the original untagged comment, as character indices.
- `ref_tags`: The list of tags for this comment in order.
- `confs`: The annotator's confidence rating for each tag. This column and the previous 3 are all lists of the same size.
- `explanation`: GPT-4o prompt-generated explanation for why tagged comment is correct. GPT-4o is not provided the win probability (WP) in generating these explanations.
- `explanation+wp`: Same as before, but GPT-4o is prompted with WP.
- `timestamp`: UNIX timestamp for whent the comment was submitted.
- `team`: In-group team corresponding to the comment &mdash; inferred from the subreddit on which the comment was made.
- `opp`: Out-group opponent team for the game.
- `username_anon`: Anonymized (integer mapped) Reddit usernames.
- `flair`: Reddit user's flair at the time of downloading the comments.
- `votes`: Voting score of the comment on the thread.
- `win_prob`: Win probability for the in-group team just before the comment was made &mdash; inferred by aligning timestamps with play-by-plays and statistics from the [`nflfastr`](https://github.com/nflverse/nflfastR/) package and [`nflverse` data](https://github.com/nflverse/nflverse-data/releases/tag/pbp).
- `gametime`: Normalized gametime at which the comment was made. 0 is start of the game, while 1 is end of the game.

The other files (`gameinfo.tsv`, `postinfo.tsv`, and `nfl_teams.csv`) are metadata files that provide information corresponding posts with games, and NFL teams.