# Attribution and reuse scope

Reference: Teng Shi, Zihua Si, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Dewei Leng, Yanan Niu and Yang Song. *UniSAR: Modeling User Transition Behaviors between Search and Recommendation.* SIGIR 2024. [arXiv](https://arxiv.org/abs/2404.09520), [ACM DOI](https://doi.org/10.1145/3626772.3657811).

The architecture and losses were independently written from the published method. Authors' source was consulted to identify implementation differences; it was not copied. The inspected upstream revision is `1a14a246066cd1aa6ee789c57ff2c7499f3d7379`. No explicit reuse license was verified for that revision. This repository does not represent an official release by the paper's authors.

Data preparation, metrics, batch handling and checkpoint utilities originated in our TRACE project and were adapted into standalone modules. The [mapping](paper-mapping.md) distinguishes these protocol choices from the paper.

KuaiSAR is credited separately: [official project](https://kuaisar.github.io/), [Zenodo release](https://zenodo.org/records/8181109), [project license](https://github.com/KuaiSAR/KuaiSAR.github.io/blob/main/LICENSE). Raw and generated datasets are excluded from Git. Model exports should be reviewed against dataset terms before a public release. Public release and software licensing remain separate project decisions.

