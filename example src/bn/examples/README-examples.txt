
Examples of Bayesian Networks for CSC242

Note that when read from an XMLBIF file, the values of Boolean random variables
are the strings "true" and "false", not the Java Boolean values TRUE and FALSE
or the primitive values true and false. Make sure to use the correct values
when specifying evidence for the inference algorithms.

- aima-alarm
  - AIMA Figure 14.2
  - Five boolean variables, 10 parameters

- aima-wet-grass
  - AIMA Figure 14.12
  - Four boolean variables, 9 parameters

- dog-problem
  - from Charniak (1991)
  - Five boolean variables, 10 parameters

- car-starts
  - from Heckerman et al. (1995)
  - 18 variables (all binary except one ternary), 121 parameters if I counted correctly

- sachs
  - from Sachs et al. (2005)
  - 11 variables, 178 parameters

- alarm
  - from Beinlich et al. (1989)
  - 37 variables with 2-4 values each (509 parameters)
  - alarm.xml created from alarm.bif by gf

- insurance
  - from Binder et al. (1997)
  - 27 variables, 984 parameters
  - insurance.xml created from insurance.bif by gf

- hailfinder
  - from Abramson at al. (1996)
  - 56 variables, 2656 parameters
  - hailfinder.xml created from hailfinder.bif by gf

- diabetes
  - from Andreassen et al. (1991)
  - 413 variables, 429409 parameters
  - note: my BIF parser apparently can't parse this one...

References

Abramson, B., J. Brown, W. Edwards, A. Murphy, and R. L. Winkler (1996).
Hailfinder: A Bayesian system for forecasting severe weather. International
Journal of Forecasting, 12(1):57-71.

Andreassen, S., R. Hovorka, J. Benn, K. G. Olesen, and E. R. Carson (1991).
A Model-based Approach to Insulin Adjustment. In Proceedings of the 3rd
Conference on Artificial Intelligence in Medicine, pp. 239-248. Springer-Verlag.

Beinlich, I. A., H. J. Suermondt, R. M. Chavez, and G. F. Cooper (1989).
The ALARM Monitoring System: A Case Study with Two Probabilistic Inference
Techniques for Belief Networks. In Proceedings of the 2nd European Conference
on Artificial Intelligence in Medicine, pp. 247-256. Springer-Verlag.

Binder, J., D. Koller, S. Russell, and K. Kanazawa (1997). Adaptive
Probabilistic Networks with Hidden Variables. Machine Learning, 29(2-3):213--244.

Charniak, E. (1991). Bayesian Networks Without Tears. AI Magazine, Winter 1991,
pp. 50--63.

Heckerman, D., J. S. Breese, K. Rommelse (1995). Decision-Theoretic Troubleshooting.
In Communications of the ACM, 38:49-57.

Sachs, K., O. Perez, D. Pe'er, D. A. Lauffenburger and G. P. Nolan (2005).
Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data.
Science, 308:523-529.
