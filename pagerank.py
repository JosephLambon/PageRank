import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    page_dist = dict() # Create dictionary for probability distribution.
    N = len(list(corpus.keys())) 

    # If a page has no outgoing links, return dist w equal probabilities.
    if not corpus[page]:
        page_dist = {item: (1/N) for item in corpus}
        return page_dist

    for item in corpus:   
        P_chooses = 0
        # Calculate P(surfer chooses each of all pages in corpus) * (1 - DAMPING)
        P_random = (1-damping_factor) / N

        if item in corpus[page]:
            # Calculate P(surfer chooses each of links on page, if each have equal P's) * DAMPING
            P_chooses = damping_factor / len(corpus[page])

        # Calculate overall probability distribution for navigating from the inputted page to all others
        P_total = P_chooses + P_random
        # Add this key value pair to dictionary
        page_dist[item] = P_total
    
    return page_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    prob_dist = {item: 0 for item in corpus} # Initialise dictionary for probability distribution
    data = []

    start_page = random.choice(list(corpus.keys())) # Choose a starting page for sampling at random
    
    for i in range(n):
        sample = transition_model(corpus, start_page, DAMPING) # Collect a sample distribution
        data.append(sample) # Append sample to 'data' array
        # Choose next start-page based off of previous sample
        options = list(sample.keys()) # Extract keys as a list (By default, .keys() extracts a view)
        weights = list(sample.values()) # Extract values as a lists
        # In the below, k=1 specifies the function to make only one choice
        start_page = random.choices(options, weights=weights, k=1)[0] # Still return the one choice in a list, so slice [0].

    #  Accumulate the probabilities from all the samples
    for sample in data:
        for item, probability in sample.items():
                prob_dist[item] += probability

    # Normalise the rankings using n rows of data
    for item in prob_dist:
        prob_dist[item] /= n # Normalise the likelihood of selection
    
    return prob_dist

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Begin by assigning each page a rank of 1/N.
    N = len(corpus)
    page_ranks = {item: (1/N) for item in corpus}

    # Iteratively calculate new rank values via formula
    while True:
        new_page_ranks = dict()
        for p in corpus:
            part1 = (1 - damping_factor)/N # First part of formula (1-d)/N
            part2 = 0 # Initialise the sigma sum (second part of formula)

            for i in corpus:
                if p in corpus[i]:
                    part2 += page_ranks[i]/len(corpus[i]) # Divide PR(i) by No. links present on page i.
                if len(corpus[i]) == 0:
                    part2 += page_ranks[i]/N # Interpret page w no links as having one for every page in corpus.

            # Multiply by damping factor
            part2 = damping_factor * part2

            PR_p = part1 + part2 # Calculate new Page Rank
            new_page_ranks[p] = PR_p # Update PR value in dictionary


        # Exit when rank values differ by less than 0.001
        delta = max(abs(new_page_ranks[p] - page_ranks[p]) for p in corpus) # delta between new PR and last iteration's
        if delta < 0.001:
            break
        page_ranks = new_page_ranks.copy()

    # Return calculated page ranks
    return page_ranks


if __name__ == "__main__":
    main()
