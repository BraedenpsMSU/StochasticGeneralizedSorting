import json

import matplotlib.pyplot as plt

with open("results.json", 'r') as data_in:
    data = json.load(data_in)


def get_queries_and_time(data, c1, funct=None, time_scale=10**9, query_scale=1000):
    x_values = sorted([int(item) for item in data[c1].keys()])
    queries_data = list(map(lambda x: list(map(lambda y: y[0], x)),
                                           map(lambda x: data[c1][x], [str(x) for x in x_values])))
    time_data = list(map(lambda x: list(map(lambda y: y[1], x)),
                         map(lambda x: data[c1][x], [str(x) for x in x_values])))
    if funct:
        queries_data = list(map(lambda x: x/query_scale, map(funct, queries_data)))
        time_data = list(map(lambda x: x/time_scale, (map(funct, time_data))))
    return x_values, queries_data, time_data


xv, q1v, t1v = get_queries_and_time(data, "1", lambda x: sum(x)/10)
_, q5v, t5v = get_queries_and_time(data, "5", lambda x: sum(x)/10)
_, q10v, t10v = get_queries_and_time(data, "10", lambda x: sum(x)/10)

_, mq1v, mt1v = get_queries_and_time(data, "1", lambda x: sorted(x)[0])
_, mq5v, mt5v = get_queries_and_time(data, "5", lambda x: sorted(x)[0])
_, mq10v, mt10v = get_queries_and_time(data, "10", lambda x: sorted(x)[0])

_, Mq1v, Mt1v = get_queries_and_time(data, "1", lambda x: sorted(x)[-1])
_, Mq5v, Mt5v = get_queries_and_time(data, "5", lambda x: sorted(x)[-1])
_, Mq10v, Mt10v = get_queries_and_time(data, "10", lambda x: sorted(x)[-1])

_, miq1v, mit1v = get_queries_and_time(data, "1", lambda x: (sorted(x)[4]+sorted(x)[5])/2)
_, miq5v, mit5v = get_queries_and_time(data, "5", lambda x: (sorted(x)[4]+sorted(x)[5])/2)
_, miq10v, mit10v = get_queries_and_time(data, "10", lambda x: (sorted(x)[4]+sorted(x)[5]))


fig, ax = plt.subplots()
ax.plot(xv, q1v, color='r', label='c=1')
ax.fill_between(xv, mq1v, Mq1v, alpha=.5, linewidth=0, color='r')
ax.plot(xv, q5v, color='g', label='c=5')
ax.fill_between(xv, mq5v, Mq5v, alpha=.5, linewidth=0, color='g')
ax.plot(xv, q10v, color='b', label='c=10')
ax.fill_between(xv, mq10v, Mq10v, alpha=.5, linewidth=0, color='b')
plt.xticks()
plt.legend()
plt.title("Number of Queries vs. Input Size")
plt.ylabel("Number of Queries (in Thousands)")
plt.xlabel("Number of vertices of input graph")
plt.savefig("queries.pdf")


fig, ax = plt.subplots()
ax.plot(xv, t1v, color='r', label='c=1')
ax.fill_between(xv, mt1v, Mt1v, alpha=.5, linewidth=0, color='r')
ax.plot(xv, t5v, color='g', label='c=5')
ax.fill_between(xv, mt5v, Mt5v, alpha=.5, linewidth=0, color='g')
ax.plot(xv, t10v, color='b', label='c=10')
ax.fill_between(xv, mt10v, Mt10v, alpha=.5, linewidth=0, color='b')
plt.legend()
plt.title("Average Execution Time vs. Input Size")
plt.ylabel("Time (seconds)")
plt.xlabel("Number of Vertices of Input Graph")
plt.savefig("time.pdf")
