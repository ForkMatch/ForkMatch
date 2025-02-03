**ForkMatch: Edge-Labelled Subgraph Matcher**

---

**Graph Data Format**

ForkMatch uses a graph format based on GSI's *.g format and is described as follows:

A file starts with `t # 0` and ends with `t # -1`.

The second line states the vertex count, edge count, vertex label maximum and edge label maximum.
For example, the line `1000 2000 1 86` corrseponds to 1000 vertices, 2000 edges, vertex label is maximum 1 and edge labels are at most 86 (inclusive).

A vertex is written as "v \<Vertex ID\> \<Label\>".

The line `v 3 1` states vertex with ID 3 has label 1.

An edge is written as "e \<Source Vertex ID\> \<Destination Vertex ID\> \<Label\>".

The line `e 3 4 2` states vertex 3 is connected vertex 4 with an edge with label 2.

Note that our system requires an undirected edge between vertices be represented as two lines in the graph text file:
A vertex 3 and 4 connected means the file must include both `e 3 4 <Label>` and `e 4 3 <Label>`

Our graph parser only works with Unix Line Endings. \r\n is not supported.

---

**Instructions**

To build ForkMatch use the build script in this repo, with `./build.sh`. It is currently configured for Turing Devices, but feel free to change the compute feature level to your target. 

Note we've only tested whether this works on Turing and Ampere devices, if there are target specific issues, please feel free to leave an issue request.

To run it use `./build/ForkMatch \<Query Graph Location\> \<Data Graph Location\>

Currently the preallocation is set high for use on RTX 8000 GPUs, but for lower end GPUs this will need to be reduced. All configuration values are found in the `environment.cuh`.

If you are running into issues of either the preallocating failing to give an allocation or running out of memory banks you can:

Reduce the size of the entire preallocation via `#define PREALLOC 45'600'000'000`

Reduce the size of the TBS allocation via `#define SOLN_PREALLOC 24'000'000'000`

Reduce the size of the output table preallocation via `#define OUTPUT_PREALLOC 17'300'000'000`

Reduce the size of the fixed size of each memory bank via `static const unsigned int CurBankSize = 1 << 20;`

<!--
**ForkMatch/ForkMatch** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
