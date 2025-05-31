# 🧩 Mefikit: A Modern Mesh Library for Scientific Simulation at CEA

## 🎯 Executive Summary

**Mefikit** is a new mesh-processing library designed with modern software
principles, safe memory handling, and high performance in mind. It is
implemented in **Rust**, a systems programming language that guarantees memory
and thread safety without garbage collection.

Born out of the limitations and frustrations with existing tools like
**MEDCoupling**, Mefikit aims to provide a **clean, robust, and interoperable
geometry core** for unstructured mesh data — suitable for both high-performance
computing (HPC) and lighter simulation workflows.

This project is an opportunity for the **CEA to reclaim strategic autonomy**,
improve code maintainability, and enable long-term collaboration and innovation
in simulation environments.

<!-- end_slide -->

## 🛠 Why a New Mesh Library?

### 🚨 MEDCoupling: Valuable Legacy, But Some Pain Points

MEDCoupling has served for mesh in memory data exchange at CEA and EdF for many
years. It has enabled simulation tools to interoperate through ICoCo, and some
Pre/Post-processing tools to build complex mesh procedurally. However, as
software practices and user expectations evolve, **some challenges have
emerged** that suggest the opportunity for **a new modern tool**.

MEDCoupling, while functional, suffers from:

#### ⚙️ Functional Issues
- **Architectural complexity**: The design is tightly coupled to the MEDFile
  format, making it difficult to isolate or evolve core mesh functionality.
  This entanglement hinders reuse and integration in modern workflows (e.g.,
  WebAssembly or Jupyter environments).
- **API overload**: Difficult to learn and use — especially for new users and
  Python developers.
- **Hidden instability**: Algorithms (e.g. intersection, projection) fail
  in corner cases, with unclear error reporting, affecting user confidence in
  automation.
- **Tight coupling to MEDFile**: Introduces I/O and format assumptions deep
  into core algorithms: groups / aggregation / merge logic scattered between
  low-level C++ and file-dependent structures.

#### 🧹 Maintenance Issues
- **Outdated codebase**: Poor encapsulation, manual memory management, no use
  of modern C++ idioms, enormous dependency on fragile SWIG code for Python
  bindings.
- **Sparse user documentation, no dev documentation**: Hard for users and
  impossible for new maintainers to onboard.
- **Platform limitations**: Not portable to ARM or modern platforms due to
  custom assembly code and tedious unsafe casting issues.
- **Complex build**: Difficult to integrate in CI/CD or Python packaging
  ecosystems. We are **not** DevOps ingeneers.

#### 🤝 Strategy and Governance Issues
- **EDF dominance**: As EdF leads development, CEA often adapts to changes
  rather than shaping direction collaboratively.
- **Loss of autonomy**: CEA cannot shape the roadmap as it lacks internal
  mastery of the code.
- **Shifting scope**: MEDCoupling has become monolithic with more and more
  dependencies and troubles to keep up with there updates, making reuse and
  integration harder, not easier.


> 🟢 **Key takeaway**: These are natural challenges in any mature codebase.
> They highlight the **need for experimentation** and **complementary tools**
> that can be more agile, modern, and modular — progressivly replacing
> the established platform.


<!-- end_slide -->

## 🚀 What Mefikit Offers

### ✨ A Clean Slate
Mefikit is designed from the ground up with the following principles:

| Principle              | Impact                                        |
|------------------------|-----------------------------------------------|
| **Minimal API**        | Easy to learn, maintain, and extend           |
| **Separation of concerns** | Geometry, metadata, and I/O are cleanly separated |
| **Correctness-first**  | Invariants and validations built into the API |
| **Zero-cost abstractions** | Safe but fast — no overhead at runtime        |

### 🧱 Solid Foundations

- **Rust Safety**: Eliminates memory leaks, use-after-free, and race
  conditions.
- **Explicit Ownership Model**: Clear semantics for shared, mutable, or
  immutable mesh views.
- **FFI-Ready**: Can expose **simple C APIs** and **Python bindings** for
  downstream use.

<!-- end_slide -->

## 🦀 Why Rust? A Strategic Choice for Scientific Software

Rust is a modern systems programming language focused on **performance**,
**correctness**, and **safety**. It has seen rapid adoption in systems,
cryptography, aerospace, and increasingly in scientific computing.

For CEA’s scientific software division, Rust represents an opportunity to
**modernize tooling** while **preserving low-level performance** and
**enforcing high-level safety**.

Researchers could spend more time writing algorithms than tracking memory
managment issues.

<!-- end_slide -->

### ✅ Advantages for Scientific Codebases

| Feature                                             | Benefit to CEA                                                          |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| 🧠 **Memory safety without GC**                     | Eliminates segfaults, leaks, and use-after-free at compile time         |
| 📚 **Strong type system**                           | Catches logic bugs early and improves code clarity                      |
| 🔁 **Zero-cost abstractions**                       | As fast as C/Fortran, but safer and easier to maintain                  |
| 🧪 **Built-in unit testing**                        | Encourages test-driven development, even for core algorithms            |
| 🔐 **Concurrency safety**                           | Prevents data races at compile time — valuable for parallel simulation  |
| 📦 **Excellent dependency ecosystem (`crates.io`)** | Reuse of solid packages for math, I/O, compression, CLI, etc.           |
| 🌐 **Interoperable via FFI**                        | Easily embeds in C, Python, or Fortran codebases                        |
| 🧩 **Modular build system (`cargo`)**               | Encourages small, reusable components (e.g., geometry, mesh, FEM logic) |

<!-- end_slide -->

### 🧪 Maturity in HPC Context

Rust is still relatively new in the HPC world, but **gaining ground rapidly**.
Mefikit targets the sweet spot of HPC-adjacent tools:

| Use Case                 | Rust Suitability        | Comments                                                   |
| ------------------------ | ----------------------- | ---------------------------------------------------------- |
| Mesh algorithms (serial) | ✅ Ideal                | Safe, fast, multithreaded and easily testable, popular in 3d rendering |
| Web assembly             | ✅ Ideal                | Native compilation with `cargo`                            |
| Python/C/C++ interop.    | ✅ Excellent            | Mature binding crates for Python/C (`maturin`, `cbindgen`) |
| File formats (MED, VTK)  | ✅ Excellent            | Mature I/O crates (`serde`, ...)                           |
| GPU & vectorization      | ⚠️ Still experimental    | Crates exist (`cust`, `wgpu`, `rayon`), not yet mainstream |
| MPI-based parallelism    | 🚧 Early adopters only  | `rsmpi` and `mpiio` exist, but lack ecosystem depth        |

➡️ **Mefikit is well-positioned** because it doesn’t need the most bleeding-edge
HPC features — yet can evolve toward them as Rust matures.

<!-- end_slide -->

### 🧩 Real-World Adoption

Rust has already been adopted in:

* **Fermilab** (High Energy Physics) – DAQ and experiment data processing
* **NASA/JPL** – Safety-critical embedded software
* **Google/Meta/AWS** – Infrastructure and secure system tools
* **Mozilla/Sentry/Cloudflare** – Networking and safe concurrency

Within France, interest in Rust is growing at INRIA, CNRS, and some HPC labs
(CEA Tech/LETI included). Supporting it in the SGLS would:

* Align with CEA’s innovation goals
* Attract young talent with modern skills
* Avoid tech debt of outdated, unsafe C++ codebases

<!-- end_slide -->

## 📦 Core Design: Ownership-Aware Mesh APIs

Mefikit introduces a three-tiered mesh abstraction:

| Type             | Purpose                          | Ownership | Use Case                       |
|------------------|----------------------------------|-----------|--------------------------------|
| `UMesh`          | Fully owned mesh structure       | Yes       | Internal use, serialization, transformation |
| `UMeshView`      | Read-only view over a mesh       | No        | FFI, safe inspection, algorithms |
| `UMeshViewMut`   | Mutable view over mesh parts     | No        | In-place transformation or assignment |

This structure enables zero-copy processing, safe parallel access, and
efficient subsetting of large meshes — **without introducing bugs or
inconsistencies**.

<!-- end_slide -->

## 🧪 Shared Coordinates and Copy-on-Write

Mefikit implements `SharedCoords`, a reference-counted coordinate storage with
on-demand copy-on-write behavior:

```rust
pub struct SharedCoords {
    pub inner: Rc<RefCell<Array2<f64>>>
}
```

This allows:
- Efficient sharing of coordinate data between meshes
- In-place geometry modifications (e.g. rigid transforms)
- Safe desynchronization when deeper changes are needed (e.g. pruning)

➡️ This kind of **granular control** is essential for hybrid workflows like
multi-physics coupling or adaptive remeshing.

<!-- end_slide -->

## 📜 Planned High-Level Algorithms

Mefikit aims to gradually offer high-level operations such as:

- Mesh aggregation/splitting by field, family, or group
- Node merging based on geometric tolerance
- Connectivity transformation
- Submesh extraction and projection
- Grouped data remapping
- Geometry-conforming boolean operations (intersection, union, cut)

🧠 Thanks to its clean core, these algorithms will be both **robust** and
**unit-tested**, unlike many unstable features in MEDCoupling.

<!-- end_slide -->

## 🔌 Interoperability Strategy

| Language | Binding Plan                    | Use Case                          |
|----------|----------------------------------|-----------------------------------|
| **Rust** | Native API                      | Algorithm development, tooling    |
| **Python** | PyO3 with NumPy interop        | Scripting, prototyping, notebooks |
| **C**    | Opaque pointer + simple FFI      | Legacy simulation codes           |

➡️ Mefikit can be embedded in any code stack and is **packaging-friendly**
(e.g., `pip`, `conda`, `cargo`).

<!-- end_slide -->

## 🏗 Development Roadmap

| Phase               | Goals                                      |
|---------------------|--------------------------------------------|
| ✅ Foundations      | Core types, ownership model, serialization, submeshing |
| 🚧 In Progress      | Field and groups assignment, views, SharedCoords  |
| 🗓 Next             | I/O bindings (MEDFile, VTK)    |
| 🚀 Milestone        | Public alpha release + Python package      |
| 🤝 Future           | Parallel versions, new complex algorithms (hexameshing, polymeshing, etc)  |

<!-- end_slide -->

## 🔑 Why It Matters for CEA

Mefikit will allow the CEA to:

- 📌 **Own its tooling**: No more dependence on EDF for feature decisions.
- 🧠 **Grow internal expertise**: Rust is readable, testable, and safer than
  C++.
- 📈 **Improve user satisfaction**: A focused and ergonomic API that Python
  users can actually enjoy.
- 🔄 **Ensure long-term maintainability**: Small, modular codebase, with modern
  CI and packaging in mind.
- ♻️ **Simplify workflows**: Replace scattered and unstable MEDCoupling features
  with clean, reliable components.

Mefikit aligns with the CEA’s mission to deliver **high-quality scientific
software**, especially in the simulation environment and mesh coupling domain.

<!-- end_slide -->

## 🤝 Proposal: A CEA-Backed Initiative

We propose that the CEA:

1. **Formally recognize Mefikit** as an internal project.
2. **Allocate time and collaborators** to help develop and validate it.
3. **Identify candidate codes** (small or prototypical) for early integration.
4. **Evaluate replacement paths** for key workflows now dependent on
   MEDCoupling.

<!-- end_slide -->

## 📢 Conclusion

Mefikit is more than a mesh library — it is a **strategic investment** in the
quality, stability, and sovereignty of our simulation infrastructure.

This is our chance to:
- Do better than the status quo,
- Take back control,
- And build something our users actually enjoy using.

Let’s support Mefikit — and take back the mesh.
