# hep_tables
 Experiments in tables

 ## Introduction

 This is prototyping accessing hierarchical data and using other backends to process it (like `servicex`, etc.).

 As a side note - this things is severly restricted by lack of a type system or type info. We need to be able to tell the difference between something that is a single item (like jet.pt()) and something that is a sequence (jet.tracks()).
 
 ## Some Notes

 - We have split the parser into two - one looks for things having to do with LINQ processes, and the other deals with things connected with expressions ("/" or ">" operations). We have to be able to call back and forth between them at the moment. Which might not be ideal. There may be a way to combine them. The problem is that the expression processor has to forward everything to the other and vice versa. Which means you have to touch both. It works, but it will be a little bit of a mess going forward, I suspect.