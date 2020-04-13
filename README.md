# hep_tables
 Experiments in tables

 ## Introduction

 This is prototyping accessing hierarchical data and using other backends to process it (like `servicex`, etc.).

 As a side note - this things is severly restricted by lack of a type system or type info. We need to be able to tell the difference between something that is a single item (like jet.pt()) and something that is a sequence (jet.tracks()).
 
 ## Some Notes

 - We have split the parser into two - one looks for things having to do with LINQ processes, and the other deals with things connected with expressions ("/" or ">" operations). We have to be able to call back and forth between them at the moment. Which might not be ideal. There may be a way to combine them. The problem is that the expression processor has to forward everything to the other and vice versa. Which means you have to touch both. It works, but it will be a little bit of a mess going forward, I suspect.

 - How should we consider parsing these things - there are math expressions and sequence expressions. We should look carefully at the DAG and make sure the two are well seperated.
 
 - When a user uses a new `histogram` function (something that comes from outside) there should be an automatic way to let the local render stuff know that a backend needs to be called for it. Right now that is hardwired into the code. And one could think of splitting some of this - as different backends could implement the same thing - so one frontend (like histogram) and then all the backends could implement it.

 - It is pretty clear that we should bring every single loop item forward as part of a tuple, just do it. That way if we need them, they are there, and we don't have to have this complex structure to go back and access them at a later time. Code will be much simpler. And the rendering code is smart enough not to try to render things that aren't needed (at least, in the xAOD backend).

 - Everythign should be typed. And we should use Sequence[xxx] to represent the sequecne, so write our own templated type
 
 - We need to monitor the "depth" we are working on in a sequence. Sequence can be nested,
   and some how we need to normalize how we think about that. As you have to nest Select
   statements the deeper you go. Currently you can go one or two in this code, this needs
   to be something general.
