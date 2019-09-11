this code contains two parts: news graph building part and recommendation tasks part

first run /src/build_news_graph.py to build the news graph

then use [Fast-TransX](https://github.com/thunlp/Fast-TransX) to get the entities' embedding

then run the recommendation tasks under /src/recommendation_tasks(currently I only upload ctr prediction task, I will release other tasks in the future)

Due to the Microsoft pricacy policies, I can only offer a toy data example