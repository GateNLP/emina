## About

Emina stands for Emergency Informativeness and Actionability, and is a tool designed to assist crisis responders and workers in filtering out unneeded internet messages.

This is the actionability component. It attempts to recognise nine categories of information:

* A: A specific resource, where the kind of need is given explicitly.
* D: Mentions of some group that's responding or aiding (e.g. volunteers, military, government, businesses, aid agencies).
* F: Threats to the general crisis response. Weather warnings, fires, military action etc.
* G: Changes in accessibility - limits to access, or other changes in how much transport can get through.
* I: Damage to infrastructure, livelihoods etc.
* J: Mention by name of the geographic areas affected
* M: Changes in the local environment (weather, hazards, etc.); e.g., a storm is intensifying
* N: General reporting about the rescue effort (from the media or the public)
* Y: Opinion or individual message (e.g. thoughts and prayers)


There's a very broad range of text and styles, especially in crisis situations, and so there's a noticeable chance that some messages will be skipped and others will be mis-included. If you find this happening, please send a sample of the data to us -- we'd love to integrate more data in order to improve performance.

## Getting started

You can get quickly started with Emina's actionability tool by putting the content to be filtered into a text file, one message per line. In this example, we'll call that messages.txt. Then, to select one kind of information, using the letters from the list above and Emina's default models (in this case, damage to infrastructure and livelihoods):

> ./filter.py --modelfilename models/actionability.I.model.sim0_45.classifier --inputfilename messages.txt

Emina's current estimate of the relevant data is filtered and printed out.

There's a bit of an overhead to starting up - around 40s on a fast machine - so it's better to send larger batches of messages at a time. 


## Development 

You can run ./filter.py --help to see the available options. 

Feature generation: embprox.py <textfile>, which expects files to have the name actionability.[letter].train

Model training: build-model.py <featurefile>

Classification: filter.py <model file> <featurefile>

Evaluation & development: see xeval.py

## License

Emina is distributed under the Creative Commons 4.0 license, CC-BY.