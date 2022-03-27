# ErCaNet (name in progres...)

This project is under construction...

## What is ErCaNet about?



# How to manage with the package
## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/08_package_a_project.html)


# Process of creating the ErCaNet (Raport section)
Below is the time schedule of work putted into the project. *Estimated*.
## Week 1
Firstly and the most important was that, we had to team up. Nextly we initiated the repo on github and made kedro project structure. We putted into `src/Project_DL/pipelines/train_model_pipeline/nodes.py` the example from `Pytorch-Lighting` website, to just test the Kedro pipline-hooks structure. We tested it on Deep Note Team Project. We logged the overview of training process with `WandB`. Nothing interesting, it was just dummy model.
## Week 2
We established the theme of our project: we want to remove the caption from the images. So we come up with the firt task that had to be done in the first place.

1.  Research of the problem in the articles.
2.  Prepare the dataset, firstly, most importantly, the training set.
3.  Prepare proper `README`, with nice raport section and, if possible, help others.

We splitted the workload among ourselves. 
## Week 3
**TBD:** Discuss the work done and establish new plan. 
