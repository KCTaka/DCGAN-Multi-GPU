# DCGAN Reflection

## Motivation
When I just heard about CNNs and image generation, I wondered how a neural network capable of generating such fine details is trained; it was a enigma for me as on how to tell the neural network which direction to update itself to improve its generation. The concept of GAN, being a tug-of-war-like training method, intrigued and enticed me to further explore how far I can bring this concept to life. I was initially introduced to the potential of the model in https://thispersondoesnotexist.com/, and properly in depth during an APS360 course at the University of Toronto. I decided to read the official original GAN paper and decided to implement it myself with a slight twist

## Problems
I've faced many problems during this project as I was writing raw pytorch and torchrun code following the official documents and tutorials (https://www.youtube.com/watch?v=-LAtx9Q6DA8). 

* The loss calculation for when training the generator is slightly different in the paper than what I'd see in other implementations

The paper says to maximize the BCELoss with 0, whereas the other implementation (including this project) minimized the BCELoss with 1. Both effectively do similar things but mathematically should be quite different. I'm not too sure as to why there is this difference, but I presume both method works

* Tough to figure whether the training was working. 

I would often test the program out many times and it would resort to unsatisfactory results and black images, and would leave me confused on whether the hyperparameter is bad, algorithm is implemented poorly/inaccurately, or there is a coding mistake. This would often lead to time consuming debugging process as it is not fast to see it in working progress.

## Outline

The paper originally describes to update the discriminator k times before updating the generator once. I implemented to update the discriminator k times before updating the generator m times. This allowed precise ratio between the discriminator and generator parameter updates. However, I assume the reason why the author did not do this is due to several other factors that can help balance the learning strength of discrminator and generator. 

!!! algorithm "Algorithm 1: Minibatch Stochastic Gradient Descent Training of GANs"
    ```text
    for number of training iterations do
        for k steps do
            sample minibatch of m noise samples {z^(1), …, z^(m)} from noise prior p_g(z)
            sample minibatch of m examples {x^(1), …, x^(m)} from data generating distribution p_data(x)
            update the discriminator by ascending its stochastic gradient:
                ∇_θd (1/m) Σ_{i=1}^m [ log D(x^(i)) + log(1 − D(G(z^(i)))) ]
        end for
        sample minibatch of m noise samples {z^(1), …, z^(m)} from noise prior p_g(z)
        update the generator by descending its stochastic gradient:
            ∇_θg (1/m) Σ_{i=1}^m [ log(1 − D(G(z^(i)))) ]
    ```

