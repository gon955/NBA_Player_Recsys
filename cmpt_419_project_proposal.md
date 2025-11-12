# CMPT 419 Project Proposal 
Members: Ivan Aceja Uy

## Role Based Recommender System for NBA Teams, Through Era Specific Clustering.

## Track Selection
Mixed between track 1 and 2 with a larger focus on track 1

## Motivation
Much discourse around basketball and sports in general revolves around the fit of a player on a team. Debates on whether a player's skill was either diminished or enhanced by a teams playstyle is commonplace among many basketball enthusiasts. Through the use of this tool one could have a more stats based approach to this discussion.

The tool aims to provide users
- A method to explore team and player archetypes per era
- Recieve player recommendations per team within the same era

## Data Source
The data was sourced from [Kaggle Datasets](https://www.kaggle.com/datasets/wyattowalsh/basketball)

The data has a wide spread of both player and team box score and advanced stats. The dataset also covers a large period of time from 1946 - Present and is constantly getting updates.

## Methodology

### Hard Filtering Through Eras
In an attempt to normalise the data in a feasible way, seasons will be bucketed into self defined eras.
- Pace and Space (2016 - Present)
- Early Spacing Adaptiaon (2008 - 2015)
- Post Hand Check / Allowed Zone Defence (starting in 2001-2002) (1999 - 2007)
- Heavy Post Scoring / Physical Defence (1990 - 1998)

When users select a team e.g 2014 Utah Jazz only players from their era of 2008 - 2015 will be recommended

### Normalisation through Era-Specific Scaling
Player and team features will be standardised within their own eras to prevent league-wide trends over eras from inflating certain stats.

### Clustering
Players and teams will be clustered based on these features

```
{
  features = [
    "pts_per_36_min","fga_per_36_min","x3pa_per_36_min","fta_per_36_min",
    
     "ast_per_36_min","tov_per_36_min",
     
     "orb_per_36_min","drb_per_36_min",
     
     "stl_per_36_min","blk_per_36_min","pf_per_36_min",
     
     "fg_percent","x3p_percent","ft_percent","e_fg_percent","ts_percent",
     
     "usg_percent","orb_percent","drb_percent","ast_percent","tov_percent",
     
     "per","ws_48","obpm","dbpm","bpm","vorp",
]
```
}

With Labels
```
{
    cluster_labels = {
    0: "Rim Protecting Big Man",
    1: "Floor General",
    2: "Spark Plug Big",
    3: "High Volume Off Ball Shooter",
    4: "High Usage Shot Creator",
    5: "MVP",
    6: "Inefficient Volume Shooter",
    7: "Star Scoring Big Man",
    8: "Inside Scoring Big Man",
    9: "Defensive Specialist Wing",
}

```
}

A similar process will occur for teams as well, but is not finished as of the moment of writing this proposal.

### Recommender

Input: Team Name, Season

#### Process
- Find teams era -> filter out players not beloning to said era
- Determine teams' archetype / playstyle
- Rank players for potenital fit of teams' playstyle

#### Output
1. Recommended player archetypes
2. List of 15 players ordered by best fit.
3. Explanation as to why: e.g this archetype of player contributes to x stat / visual aids through data visualisation

## Personal Motivations
Being a solo project, this project meets my own individual incentives of leveraging my experience with machine learning and applying it to one of my areas of interest: basketball statistics. Another motivation of mine with this project is to improve my skills surrounding front end development as it is not an area I have done much work in. Lastly, having this tool be presentable through the front end work would allow me to display it on my personal portfolio

## Human Centered / Data Centered Aspects
### Human Centered
The tool aims to be human centered through two main avenues. The first being using visualisations of data to make the result more interpretable to users. The other avenue is making the cluster labels interpretable by most people, but also showing which players best represent each cluster.

### Data Centered
The large period of time in which the data used in this project covers leads to a lot of normalisation decisions. Another factor could be attempting to uncover the impact of specific entries e.g 2008 Lebron James and seeing how that shifts decision boundaries.

## Limitations / Future Work
This project does not take into account financial/contractual constraints of each team as that data was not easily sourced / has to be scraped.
    
