# License Attribution for `pase` Repository

The project contains original work (Cisco Systems, Inc. and its affiliates) under the Apache License 2.0 and incorporates third-party components under their respective open-source licenses. Each file includes a header indicating its origin and license.

---
## 1. Original Implementation (Apache-2.0)
Files authored by Cisco Systems, Inc. and its affiliates include the header:
```
# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
```
These files are licensed under the Apache License, Version 2.0 (see https://www.apache.org/licenses/LICENSE-2.0).

Cisco-authored files:
- `pase/models/pase.py`
- `pase/models/wavlm/__init__.py`
- `pase/models/wavlm/feature_extractor.py`
- `pase/models/vocoder/wavlmdec.py`
- `pase/models/vocoder/wavlmdec_dual.py`
- `pase/inference/inference.py`
- `pase/inference/infer_dewavlm.py`
- `pase/inference/infer_vocoder.py`
- `pase/inference/infer_vocoder_dual.py`
- `pase/loaders/dataloader.py`
- `pase/loaders/dataloader_clean.py`

- `stupase/models/stupase.py`
- `pase/models/wavlm/__init__.py`
- `pase/models/wavlm/feature_extractor.py`
- `stupase/models/mel/feature_extractor.py`
- `stupase/inference/inference.py`
- `stupase/inference/infer_dewavlm.py`
- `stupase/inference/infer_flow.py`
- `stupase/inference/infermelvocoder.py`
- `pase/loaders/dataloader.py`
- `pase/loaders/dataloader_clean.py`


(If additional Cisco-authored `.py` files are added later, they should retain the same Apache-2.0 header.)

---
## 2. Third-Party Components
See [NOTICE](NOTICE) for third-party components and their respective licenses.

---
## 3. Overall Project Licensing
Original Cisco-authored portions are under Apache License 2.0. Third-party components retain their original licenses as stated. Redistribution and use must comply with each respective license.

---
## 4. How to Add New Files
When adding new source files:
- For Cisco-authored code: include the standard Apache-2.0 header.
- For third-party code: preserve original header and add the license file if not already present.
- Update `LICENSE.md` and `NOTICE` files accordingly.

---
If any discrepancies are found, please open an issue or update the relevant headers.
