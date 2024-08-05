# pylint: disable=no-name-in-module

from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt5.QtCore import QRegExp


class PythonHighlighter(QSyntaxHighlighter):  # pylint: disable=too-few-public-methods
    """
    Adds python syntax highlighting.
    """

    def __init__(self, document):
        super().__init__(document)

        # Define the formatting for keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("blue"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "\\bclass\\b", "\\bdef\\b", "\\bif\\b", "\\belse\\b", "\\belif\\b",
            "\\bwhile\\b", "\\bfor\\b", "\\btry\\b", "\\bexcept\\b", "\\bfinally\\b",
            "\\bwith\\b", "\\bas\\b", "\\bimport\\b", "\\bfrom\\b", "\\breturn\\b",
            "\\bpass\\b", "\\bbreak\\b", "\\bcontinue\\b", "\\braise\\b"
        ]
        self.highlighting_rules = [(QRegExp(pattern), keyword_format) for pattern in keywords]

        # Define the formatting for strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("green"))
        self.highlighting_rules.append((QRegExp("\".*\""), string_format))
        self.highlighting_rules.append((QRegExp("\'.*\'"), string_format))

        # Define the formatting for comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("red"))
        self.highlighting_rules.append((QRegExp("#[^\n]*"), comment_format))

    def highlightBlock(self, text):  # pylint: disable=invalid-name
        """
        Highlight a specific block of text.

        :param text: Text to highlight.
        """
        for pattern, form in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, form)
                index = expression.indexIn(text, index + length)
