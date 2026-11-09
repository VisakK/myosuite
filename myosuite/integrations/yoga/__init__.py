# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle-driven yoga imitation on a subject-scaled MyoFullBody.

Pipeline (see ``README.md`` in this package):

1. :mod:`.reference_io` imports curated yoga references (ProtoMotions ``.motion``
   files on the performer's own 24-body skeleton) without needing ProtoMotions.
2. :mod:`.subject_body` measures the performer's skeleton against MyoFullBody and
   builds a MyoFullBody spec scaled to her segment lengths and masses, with the
   muscle length ranges rescaled to the new geometry.
3. :mod:`.retarget` solves a contact-aware inverse kinematics problem per frame
   on the scaled model and writes MuscleMimic-format motion clips.
4. :mod:`.task` registers the mjlab (GPU) training task that tracks those clips
   with muscles.

The motion data is derived from the MOYO capture and AMASS-licensed fits; it is
never written inside this package.
"""
