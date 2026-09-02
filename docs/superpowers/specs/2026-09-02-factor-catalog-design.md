# Factor Catalog Design

## Goal

Give reusable factors stable, versioned identity plus dated evidence so dependencies, PIT semantics, preprocessing, implementation identity, and lifecycle are inspectable without reading implementation code.

## Design

`FactorSpec` is immutable identity metadata. `FactorEvidenceSummary` is a small dated evidence record. `FactorCatalog` registers exact `(factor_id, version)` pairs, rejects duplicates, and serializes to a platform-owned schema.

Raw factor values, model objects, provider objects, and full tear sheets remain outside the catalog. Alphalens/RQFactor-style analysis can feed evidence after normalization, but neither owns factor identity.
