import sys
import json
import numpy as np
import web
from web import form
from urllib.request import urlretrieve

from evaluate_mobilenet import predict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def mean(score_dist):
    scores = np.array([1,2,3,4,5,6,7,8,9,10])
    return (score_dist * scores).sum()

def std(score_dist):
    scores = np.array([1,2,3,4,5,6,7,8,9,10])
    u = mean(score_dist)
    return np.sqrt(np.sum(((scores - u) ** 2) * score_dist))

def download(url):
    file, _ = urlretrieve(url)
    return file


render = web.template.render('templates/')

urls = ('/', 'index')
app = web.application(urls, globals())

myform = form.Form(form.Textarea('urls')) 

class index: 
    def GET(self): 
        form = myform()
        # make sure you create a copy of the form by calling it (line above)
        # Otherwise changes will appear globally
        return render.formtest(form)

    def POST(self):
        data = web.input(urls='')
        text = data.urls
        urls = []
        for url in text.split('\n'):
            url = url.strip()
            if url.startswith('http'):
                urls.append(url)
        paths = list(map(download, urls))
        score_dists = predict(paths)
        means = [ round(mean(score_dist), 2) for score_dist in score_dists ]
        stds = [ round(std(score_dist), 2) for score_dist in score_dists ]
        indexes = sorted(range(len(means)), key=lambda i: means[i] * -1)
        return render.evaluate(indexes, urls, score_dists, means, stds)

if __name__=="__main__":
    web.internalerror = web.debugerror
    score_dist = predict(['./images/test.jpg'])
    print(score_dist)
    print(mean(score_dist))
    print(std(score_dist))
    app.run()

