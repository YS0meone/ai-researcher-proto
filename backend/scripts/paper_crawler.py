import arxiv

# Construct the default API client.
client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = "quantum",
  max_results = 10,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

results = client.results(search)

# `results` is a generator; you can iterate over its elements one by one...
for r in client.results(search):
  print(r.title)
# ...or exhaust it into a list. Careful: this is slow for large results sets.
all_results = list(results)
print([r.title for r in all_results])

# For advanced query syntax documentation, see the arXiv API User Manual:
# https://arxiv.org/help/api/user-manual#query_details
search = arxiv.Search(query = "au:del_maestro AND ti:checkerboard")
first_result = next(client.results(search))
print(first_result)

# Search for the paper with ID "1605.08386v1"
search_by_id = arxiv.Search(id_list=["1605.08386v1"])
# Reuse client to fetch the paper, then print its title.
first_result = next(client.results(search))
print(first_result.title)


# for more information, check https://pypi.org/project/arxiv/