"""The G1 contact model shared by the scripts here: scene.xml (the Menagerie g1_mjx scene with the
floor pairs only) through cusadi2.contact, with the Menagerie loader's repaired Pinocchio model."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "scene.xml")
sys.path.insert(0, os.path.join(HERE, "..", "menagerie_bench"))


def load():
    """cusadi2.contact.Model of scene.xml. Pinocchio's MJCF parser rejects the scene's <keyframe>
    ("Keyframe size does not match model size") and builds the robot twice when the scene adds a
    second <worldbody> (the floor) to the included robot's, so it parses a copy without either,
    written next to scene.xml so the relative <include> and meshdir resolve."""
    import re
    import model as menagerie
    from cusadi2 import contact
    m = menagerie.load_mujoco(XML)
    pin_xml = os.path.join(HERE, "_scene_pin.xml")
    with open(pin_xml, "w") as f:
        f.write(re.sub(r"<(keyframe|worldbody)>.*?</\1>", "", open(XML).read(), flags=re.S))
    try:
        pm = menagerie.load_pinocchio(pin_xml, m=m)
    finally:
        os.remove(pin_xml)
    return contact.Model(XML, pmodel=pm)


def disable_contacts(m):
    """The contact-free model for the smooth gate and the phase-1 premise benchmark (same flags as the
    Menagerie campaign's MJX / warp baselines)."""
    import mujoco
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    m.opt.noslip_iterations = 0
    return m
