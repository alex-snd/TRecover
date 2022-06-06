---
title: " "
---
!!! important "Abstract"
    The main goal of the project is to explore the possibility of using the Transformer neural network to “read” 
    meaningful text in columns for the English language.

For a clearer explanation of the problem, let me introduce the following concepts.

Let $A$ be the alphabet of the English language, the letters of which are arranged in natural order. The 
$\alpha + \gamma = \beta \,$ notation will be understood as the modular addition of letter numbers from 
$\{ 0, \, 1, \, 2, \, ... \, , \, |A|- 1 \}$ modulo $|A|$, where $\alpha, \, \gamma, \, \beta \in A$. 
That is, when encrypting plaintext, we will identify letters with their numbers in the alphabet $A$.

Let's consider a simple example in which only two letters $\gamma', \, \gamma'' \in A$ can be used as keystream 
$\bar{\gamma}=(\gamma_1, \, \gamma_2, \, … \, , \, \gamma_n)$  values, that is, the ciphertext 
$\bar{\beta}=(\beta_1, \, \beta_2, \, … \, , \, \beta_n)$ formation equation has the form 
$\alpha_i + \gamma_i = \beta_i, \; i \in \\{1, \, 2, \, … \, , \, n\\}$, where $n$ is the length of the message. 
In this case, recovering the plaintext $\bar{\alpha}=(\alpha_1, \, \alpha_2, \, ... \, , \, \alpha_n)$  by ciphertext
$\bar{\beta}$  isn't difficult. Indeed, we will make up the columns 
$\bar{\nabla}=(\bar{\nabla_1}, \, \bar{\nabla_2}, \, ... \, , \, \bar{\nabla_n})$ according to the known ciphertext 
$\bar{\beta}$, where $\bar{\nabla_i}=(\beta_i-\gamma', \, \beta_i-\gamma''), \; i \in \\{1, \, 2, \, ... \, , \, n\\}$:

<div align="center">
    <table align="center">
       <tr align="center"><td>\(\beta_1-\gamma'\)</td><td>\(\beta_2-\gamma'\)</td><td>\(\beta_3-\gamma'\)</td><td>\(...\)</td><td>\(\beta_n-\gamma'\)</td></tr>
       <tr align="center"><td>\(\beta_1-\gamma''\)</td><td>\(\beta_2-\gamma''\)</td><td>\(\beta_3-\gamma''\)</td><td>\(...\)</td><td>\(\beta_n-\gamma''\)</td></tr>
    </table>
</div>

Obviously, each column $\bar{\nabla_i}$ consists of unique values and contains one letter $\alpha_i$ of the plaintext 
$\bar{\alpha}$. You can try to recover this text using its redundancy. Without going into details, here is one example 
for “reading” in columns:

<div align="center">
    <table align="center">
       <tr align="center"><td>\(c\)</td><td>\(d\)</td><td>\(l\)</td><td>\(p\)</td><td>\(q\)</td><td>\(o\)</td><td>\(k\)</td><td>\(u\)</td><td>\(a\)</td><td>\(x\)</td><td>\(h\)</td><td>\(g\)</td></tr>
       <tr align="center"><td>\(y\)</td><td>\(r\)</td><td>\(y\)</td><td>\(w\)</td><td>\(t\)</td><td>\(j\)</td><td>\(g\)</td><td>\(r\)</td><td>\(b\)</td><td>\(p\)</td><td>\(m\)</td><td>\(y\)</td></tr>
    </table>
</div>
??? help "Have you read the word ..."

    Have you read the word ***cryptography*** 😉?

!!! info ""

    If four letters were used for encryption, then the column depth would be equal to four. And if all the letters were 
    used, then the depth of the columns would be equal to $|A|$ and in this case you can read any text in them.

Let's take a closer look at an example in which all letters from the alphabet $A$ can be used as keystream 
$\bar{\gamma}$ values. In this case, there are also certain approaches for making up the columns $\bar{\nabla}$.
One of them is that in each column $\bar{\nabla_i}=(\alpha_i^{(1)}, \, \alpha_i^{(2)}, \, ... \, , \, \alpha_i^{(|A|)})$
the order of possible plaintext letters $\alpha_i^{(j)} \in A$ is determined by decreasing (more precisely, not increasing) their probabilities:
$$ P(\alpha_i^{(j)} \, | \, \beta_i) = \frac{P(\alpha_i^{(j)}, \,  \beta_i)}{P(\beta_i)}=\frac{\phi(\alpha_i^{(j)}) \cdot \varphi(\beta_i-\alpha_i^{(j)})}{\sum_{\alpha’ \in A} \phi(\alpha’) \cdot \varphi(\beta_i-\alpha’)},$$
$$ i \in \\{1, \, 2, \, ... \, , \, n\\\}, \; j \in \\{1, \, 2, \, ... \, , \, |A|\\},$$
with a known fixed letter $\beta_i \in A$ of the ciphertext $\bar{\beta}$. Here $\phi(\alpha), \; \alpha \in A$ is
probability distribution of letters of meaningful texts for the alphabet $A, \; \varphi (\gamma), \, \gamma \in A$ 
is probability distribution of the $\bar{\gamma}$ keystream values. 

!!! info ""

    Also, for a more accurate ordering of letters in columns, their probabilities are calculated based on n-grams.

The depth $\bar{h}=(h_1, \, h_2, \, ... \, , \, h_n)$  of the columns 
$\bar{\nabla}=(\bar{\nabla_1}, \, \bar{\nabla_2}, \, ... \, , \, \bar{\nabla_n})$ can be limited using a pre-selected 
value of the $\epsilon \in (0, 1]$ parameter:
$$ \begin{aligned} h_i=max\\{\ell \in \\{ 1, \, 2, \, ... \, , \, |A|\\} : \sum_{j=1}^{\ell} P(\alpha_i^{(j)} \, | \, \beta_i) \le \epsilon \\}. \end{aligned}$$

The critical depth of the columns $\hat{h}$, at which it is possible to unambiguously determine the original plaintext,
is calculated by the formula:
$$ \begin{aligned} \hat{h}=|A|^{1 - H(A)}, \end{aligned}$$
where $H(A)$ is entropy of a language with alphabet $A$. 

!!! info ""

    For English, $\hat{h} \approx 13$.