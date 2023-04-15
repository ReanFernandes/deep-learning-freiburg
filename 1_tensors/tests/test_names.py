import re
from pathlib import Path


def test_names():
    """Test if the members.txt file has been filled correctly."""
    should_look_like = "name 1; mail address 1; ilias username 1"

    # epic mail regex
    RE_MAIL = re.compile(
        r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21'
        r'\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9]'
        r'(?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4]'
        r'[0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\['
        r'\x01-\x09\x0b\x0c\x0e-\x7f])+)])')

    # ilias username regex
    RE_ILIAS = re.compile('^[a-zA-Z]+[0-9]+$')

    # read members file
    file = Path("members.txt")
    assert file.is_file(), "members.txt does not exist."
    content_lines = file.read_text(encoding="utf8").splitlines()

    # parse members file
    num_members = 0
    for line in content_lines:
        if line.strip() == "":
            continue
        num_members += 1
        split_sep = line.split(";")
        assert len(split_sep) == 3, f"could not understand line, should look like this:\n{should_look_like}\n"\
                                    f"but is:\n{line}\n({len(split_sep) - 1} semicolons instead of 2)"
        name, mail, ilias = [item.strip() for item in split_sep]
        # check for RFC 5322 compliant mail
        assert RE_MAIL.match(mail), f"this is not an email address: {mail}"

        # check for ilias username
        assert RE_ILIAS.match(ilias), f"this is not a valid ilias username: {ilias}. It should be some lowercase "\
                                      f"letters followed by some numbers"


if __name__ == '__main__':
    test_names()
    print("Test complete.")
