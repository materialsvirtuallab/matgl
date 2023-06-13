"""
Pyinvoke tasks.py file for automating releases and admin stuff.
"""

from __future__ import annotations

import glob
import json
import os
import re

import requests
from invoke import task
from monty.os import cd

import matgl

NEW_VER = matgl.__version__


@task
def make_doc(ctx):
    """
    This new version requires markdown builder.

        pip install sphinx-markdown-builder

    Adding the following to conf.py

        extensions = [
            'sphinx_markdown_builder'
        ]

    Build markdown files with sphinx-build command

        sphinx-build -M markdown ./ build
    """
    with cd("docs_src"):
        ctx.run("touch index.md")
        ctx.run("rm matgl.*.rst", warn=True)
        ctx.run("sphinx-apidoc -P -M -d 6 -o . -f ../matgl")
        ctx.run("rm matgl.*tests*.rst", warn=True)
        for f in glob.glob("*.rst"):
            if f.startswith("matgl") and f.endswith("rst"):
                newoutput = []
                suboutput = []
                subpackage = False
                with open(f) as fid:
                    for line in fid:
                        clean = line.strip()
                        if clean == "Subpackages":
                            subpackage = True
                        if not subpackage and not clean.endswith("tests"):
                            newoutput.append(line)
                        else:
                            if not clean.endswith("tests"):
                                suboutput.append(line)
                            if clean.startswith("matgl") and not clean.endswith("tests"):
                                newoutput.extend(suboutput)
                                subpackage = False
                                suboutput = []

                with open(f, "w") as fid:
                    fid.write("".join(newoutput))
        ctx.run("rm ../docs/matgl*.html", warn=True)
        # ctx.run("sphinx-build -b html . ../docs")  # HTML building.
        ctx.run("sphinx-build -M markdown . ../docs")
        ctx.run("rm matgl*.rst", warn=True)

    with cd("docs"):

        ctx.run("mv markdown/matgl*.md .")
        for fn in glob.glob("matgl.*.md"):
            with open(fn, "rt") as f:
                contents = f.read()
            with open(fn, "wt") as f:
                f.write(f"---\nlayout: default\ntitle: Home\nnav_exclude: true\n---\n\n" + contents)

        with open("matgl.md", "rt") as f:
            contents = f.read()
        with open("matgl.md", "wt") as f:
            f.write(f"---\nlayout: default\ntitle: API Documentation\nnav_order: 4\n---\n\n" + contents)

        for d in (".doctrees", "_sources", "static", "markdown", "modules*"):
            ctx.run(f"rm -r {d}", warn=True)

        ctx.run("mv _static static")
        ctx.run("sed -i'.orig' -e 's/_static/static/g' matgl*.html")
        ctx.run("rm index.html", warn=True)
        ctx.run("cp ../*.md .")
        ctx.run(f"mv README.md index.md")
        ctx.run("rm -rf *.orig _site doctrees", warn=True)

        with open("index.md", "rt") as f:
            index = f.read()
        with open("index.md", "wt") as f:
            f.write("---\nlayout: default\ntitle: Home\nnav_order: 1\n---\n\n" + index)


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def release_github(ctx):
    with open("changes.md") as f:
        contents = f.read()
    toks = re.split(r"\#+", contents)
    desc = toks[1].strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "main",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/matgl/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)


@task
def release(ctx, notest=False):
    ctx.run("rm -r dist build matgl.egg-info", warn=True)
    if not notest:
        ctx.run("pytest matgl")
    publish(ctx)
    release_github(ctx)
