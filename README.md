# Estimate and cluster sentiment in tweets

## Problem

## Data

## Methodology

## Results

### Model Predictions

<p align="center">   
  <img width="350" height="350"src="https://github.com/vb690/tweets_clustering/blob/master/results/figures/predicted.png">
</p> 

**Cluster Negative**
```
* I was just very surprised it took over mins never experienced anything like that on number

* And no offers to provide us with a hotel or anything for a flight that is over hours away

* Unbelievable that i can not even wait on hold to speak to a human being to resolve my issue the system simply hangs up
```

**Cluster Neutral**
```
* I have their name boarding pass was in there too i think they might really need this any ideas

* Flight taxis at prior to flight to http

* Jetblue our fleet on fleek http i have to refrain what i want to say
```

**Cluster Positive**
```
* Haha thanks for the explanation

* Your crew on tonight was outstanding god bless them and the medically trained passengers on board

* I always look forward to jb rt looking forward to welcoming you onboard
```

### K-Means Clustering

<p align="center">   
  <img width="350" height="350"src="https://github.com/vb690/tweets_clustering/blob/master/results/figures/kmeans.png">
</p> 

**Cluster 0**

Negative : 0.106  
Neutral : 0.02  
Positive : 0.873  
```
* Glad it was finally resolved too too bad i ca get a free voucher to go with mine so i can have a friend travel next time

* Thanks to karen salisbury at iah for amazing customer service found my daughter bag lost on made her day

* Thank you for leaving my bag in houston despite what your system says i was definitely on the flight
```

**Cluster 1**

Negative : 1.0  
```
* I would be on my way but wether has delayed yet love the charlotte ice storm good luck with mad customers

* She missed her uncle funeral and you hope they can find another flight that very considerate of you

* This is pathetic customer service why have bag tags barcodes and computers if your response is not yet located
```

**Cluster 2**

Negative : 0.988  
Neutral : 0.002  
Positive : 0.01  
```
* Why cancelled flight on amp flights now hrs longer amp have layovers too late flight to find a reasonably priced alternative

* Unable to locate baggage this is frustrating all my sons clothes and needs in it zfv yyz usairways baggagelost

* Is unfriendly screw family that hates kids and moms now waiting on pray its better fyvrfn to agent error and tickt
```

**Cluster 3**

Negative : 0.999  
Positive : 0.001  
```
* I had a sjc to jfk departing tonight i have to now change my whole schedule can we get a sort of refund for this

* Believe me i understand flight was originally booked for sunday flight was cancelled flighted and rescheduled for today

* My flight is cancelled flightled and we can only call in but you are accepting calls how do i reschedule my flight
```

**Cluster 4**

Negative : 0.023  
Neutral : 0.708  
Positive : 0.269  
```
* Did you pick the winners for the destinationdragons flightr maybeijustlost whyyounoloveme

* What your current checked bag price policy i fly this coming week and could find anything definitive

* To start flights from washington reagan to nantucket between avgeek
```

**Cluster 5**

Negative : 0.37  
Neutral : 0.625  
Positive : 0.005  
```
* Hi there flight from dallas just cancelled flightled going to la can u pls help rebook me

* Is the new time confirmed or it may get cancelled flightled traveling with kids need to be certain thx

* Does virgin america fly direct from seattle to nyc or boston
```


### HDBSCAN Clustering

<p align="center">   
  <img width="350" height="350"src="https://github.com/vb690/tweets_clustering/blob/master/results/figures/hdbscan.png">
</p> 

**Cluster Noise**

Negative : 0.644  
Neutral : 0.201  
Positive : 0.155  
```
* why do you tell them to update the boards

* there was also not one single person at the counter answering questions for our plane full of confused people no staff at all

* comin in clutch and sending me to charlotte then home i u except for wayne u a real g thankjesus thankme
```

**Cluster 0**

Negative : 0.193  
Neutral : 0.089  
Positive : 0.719  
```
* can you tell me what that means i work and just need an estimated time of arrival please i need my laptop for work thanks

* your website is fucked up now instead of a flight i should ve had yesterday it s today thanks

* thanks
```

**Cluster 1**

Neutral : 0.027  
Positive : 0.973  
```
* thank you

* thank you

* ok thank you
```

**Cluster 2**

Negative : 0.991  
Neutral : 0.005  
Positive : 0.004  
```
* thanks for ruining my wedding anniversary flight from ewr to rdu is delayed by over hours and will reach home the nxt day

* omg answer your phone i have been on hold for hours cancelled flighted flights suck

* can you guys please give an update been sitting on tarmac on flight yet the website still says it on time
```

**Cluster 3**

Negative : 0.648  
Neutral : 0.206  
Positive : 0.147  
```
* tells me paid amp confirmed flight will now cost me cash only airport hours before my flight ripoff nosupport

* decisions decisions we love for you to try our service we offer status match too http

* hour delay nothing says sorry like a voucher missing time with family family precioustime
```


### Affinity Propagation Clustering

<p align="center">   
  <img width="350" height="350"src="https://github.com/vb690/tweets_clustering/blob/master/results/figures/affinity_propagation.png">
</p> 


**Cluster 0**

Negative : 0.224  
Neutral : 0.387  
Positive : 0.388  
```
* Trying to reset my password email never arrives help

* Would love tix for velour every contest and still have won

* I read this over a nice glass of wine right before i dig my heels into my next novel maybe travel is just what i need
```

**Cluster 1**

Negative : 0.963  
Neutral : 0.025  
Positive : 0.012  
```
* We been seating for inside flight at iad delayed we only been offered water amp cookies in business class failed

* Is horrible they lost our carseat and expect us to use a loner carseat safety regulations say it illegal to use a used car seat

* We still need help hung up on twice customer service rep said there is no way to help between the gate rep and phone rep
```

**Cluster 2**

Negative : 0.537  
Neutral : 0.291  
Positive : 0.173  
```
* Jetblue our fleet on fleek http foh

* Volumes profit up http aviation aircargo http

* Rt new airline expected to make its way to mem http
```


### Agglomerative Clustering

<p align="center">   
  <img width="350" height="350"src="https://github.com/vb690/tweets_clustering/blob/master/results/figures/agglomerative.png">
</p> 

**Cluster 0**

Negative : 0.157  
Neutral : 0.098  
Positive : 0.744  
```
* Thank you

* Thank you for responding so quickly with a helpful tool

* I might look into that my wife travels much more than i do could we both use the membership
```

**Cluster 1**

Negative : 0.972  
Neutral : 0.02  
Positive : 0.008  
```
* What are the odds my bag was picked up by someone not me while in another country because it says it has been located yet

* The customer service today is unsat flight cnx not notified called for hours and the phone line does not even let me hold

* Airport and extra nights for hotels and not once have i heard of anything from your embarrassing airline saying that you will
```

**Cluster 2**

Negative : 0.055  
Neutral : 0.654  
Positive : 0.291  
```
* If i had my tux it be a date umosaicmecrazy http

* I like the inflight snacks i flying with you guys on jvmchat

* This looks like jfk had a great pic of the oneworld jets lined up on the opposite side air berlin lan and aa
```

**Cluster 3**

Negative : 0.416  
Neutral : 0.573  
Positive : 0.011  
```
* Does she need to complain on twitter for the refund or is it

* Easy fix let the business select actually board then board the

* How do i stop getting credit card apps i already have a card
```

**Cluster 4**

Negative : 0.999  
Neutral : 0.001  
```
* The best is your message saying to use website and your website is saying you need to call if you do answer hardtodo

* I think it safe to say that after years of loyal jetblue flying we are officially done byebyejetblue

* Now we have to wait and wait to get it straight i fly you for personal and business but now not so sure bebetter
```

**Cluster 5**

Negative : 0.972  
Neutral : 0.019  
Positive : 0.009  
```
* Two hours on the plane at the gate this is undignified behavior for you and the apology isn t enough freedrinkcoupons

* Delayed hrs flight finally board the plane sit half hour amp crew is at their hrs limit amp we deplane unacceptable

* You have a company policy that refuses employees to speak to other employees over the phone interesting
```
