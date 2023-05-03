|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Анализ смещения распределения в задаче контрастного распределения.
    :Тип научной работы: M1P
    :Автор: Лидия Сергеевна Троешестова
    :Научный руководитель: кандидат физико-математических наук, Исаченко Роман Владимирович

Abstract
========

Recently contrastive learning has regained popularity as a self-supervised representation learning technique. It involves comparing positive (similar) and negative (dissimilar) pairs of samples to learn representations without labels. However, false negative and false positive errors in sampling lead to the loss function bias. This paper analyzes various ways to eliminate these biases. Based on the fully-supervised case, we develop debiased contrastive models that account for same-label datapoints without requiring knowledge of true labels, and explore their properties. Using the debiased representations, we measure accuracy of predictions in the classification task. The experiments are carried out on the CIFAR10 dataset, demonstrating the applicability and robustness of the proposed method in scenarios where extensive labeling is expensive or not feasible.

Software modules developed as part of the study
======================================================
1. A python package *DebiasedPos* with all implementation `here <https://github.com/intsystems/2023-Project-123/tree/master/code>`_.
2. A code with all experiment visualisation `here <https://github.com/intsystems/2023-Project-123/blob/master/code/experiments.ipynb>`_. View in `colab <https://colab.research.google.com/drive/1ZwFs8Re9bQdgzQxNsXU6yrV31C9SW8D-?usp=sharing>`_.

Acknowledgements
======================================================
Inspired by [chingyaoc/DCL](https://github.com/chingyaoc/DCL).
