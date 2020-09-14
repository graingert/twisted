# -*- test-case-name: twisted.cred.test.test_cred -*-
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

"""
Basic credential checkers

@var ANONYMOUS: An empty tuple used to represent the anonymous avatar ID.
"""


import os

from zope.interface import implementer, Interface, Attribute

from twisted.logger import Logger
from twisted.internet import defer
from twisted.python import failure
from twisted.cred import error, credentials


class ICredentialsChecker(Interface):
    """
    An object that can check sub-interfaces of L{ICredentials}.
    """

    credentialInterfaces = Attribute(
        (
            "A list of sub-interfaces of L{ICredentials} which specifies which I "
            "may check."
        )
    )

    def requestAvatarId(credentials):
        """
        Validate credentials and produce an avatar ID.

        @param credentials: something which implements one of the interfaces in
        C{credentialInterfaces}.

        @return: a L{Deferred} which will fire with a L{bytes} that identifies
        an avatar, an empty tuple to specify an authenticated anonymous user
        (provided as L{twisted.cred.checkers.ANONYMOUS}) or fail with
        L{UnauthorizedLogin}. Alternatively, return the result itself.

        @see: L{twisted.cred.credentials}
        """


# A note on anonymity - We do not want None as the value for anonymous
# because it is too easy to accidentally return it.  We do not want the
# empty string, because it is too easy to mistype a password file.  For
# example, an .htpasswd file may contain the lines: ['hello:asdf',
# 'world:asdf', 'goodbye', ':world'].  This misconfiguration will have an
# ill effect in any case, but accidentally granting anonymous access is a
# worse failure mode than simply granting access to an untypeable
# username.  We do not want an instance of 'object', because that would
# create potential problems with persistence.

ANONYMOUS = ()


@implementer(ICredentialsChecker)
class AllowAnonymousAccess:
    """
    A credentials checker that unconditionally grants anonymous access.

    @cvar credentialInterfaces: Tuple containing L{IAnonymous}.
    """

    credentialInterfaces = (credentials.IAnonymous,)

    def requestAvatarId(self, credentials):
        """
        Succeed with the L{ANONYMOUS} avatar ID.

        @return: L{Deferred} that fires with L{twisted.cred.checkers.ANONYMOUS}
        """
        return defer.succeed(ANONYMOUS)


@implementer(ICredentialsChecker)
class InMemoryUsernamePasswordDatabaseDontUse:
    """
    An extremely simple credentials checker.

    This is only of use in one-off test programs or examples which don't
    want to focus too much on how credentials are verified.

    You really don't want to use this for anything else.  It is, at best, a
    toy.  If you need a simple credentials checker for a real application,
    see L{FilePasswordDB}.

    @cvar credentialInterfaces: Tuple of L{IUsernamePassword} and
    L{IUsernameHashedPassword}.

    @ivar users: Mapping of usernames to passwords.
    @type users: L{dict} mapping L{bytes} to L{bytes}
    """

    credentialInterfaces = (
        credentials.IUsernamePassword,
        credentials.IUsernameHashedPassword,
    )

    def __init__(self, **users):
        """
        Initialize the in-memory database.

        For example::

            db = InMemoryUsernamePasswordDatabaseDontUse(
                user1=b'sesame',
                user2=b'hunter2',
            )

        @param users: Usernames and passwords to seed the database with.
        Each username given as a keyword is encoded to L{bytes} as ASCII.
        Passwords must be given as L{bytes}.
        @type users: L{dict} of L{str} to L{bytes}
        """
        self.users = {x.encode("ascii"): y for x, y in users.items()}

    def addUser(self, username, password):
        """
        Set a user's password.

        @param username: Name of the user.
        @type username: L{bytes}

        @param password: Password to associate with the username.
        @type password: L{bytes}
        """
        self.users[username] = password

    def _cbPasswordMatch(self, matched, username):
        if matched:
            return username
        else:
            return failure.Failure(error.UnauthorizedLogin())

    def requestAvatarId(self, credentials):
        if credentials.username in self.users:
            return defer.maybeDeferred(
                credentials.checkPassword, self.users[credentials.username]
            ).addCallback(self._cbPasswordMatch, credentials.username)
        else:
            return defer.fail(error.UnauthorizedLogin())


@implementer(ICredentialsChecker)
class FilePasswordDB:
    """
    A file-based, text-based username/password database.

    Records in the datafile for this class are delimited by a particular
    string.  The username appears in a fixed field of the columns delimited
    by this string, as does the password.  Both fields are specifiable.  If
    the passwords are not stored plaintext, a hash function must be supplied
    to convert plaintext passwords to the form stored on disk and this
    CredentialsChecker will only be able to check L{IUsernamePassword}
    credentials.  If the passwords are stored plaintext,
    L{IUsernameHashedPassword} credentials will be checkable as well.
    """

    cache = False
    _credCache = None
    _cacheTimestamp = 0
    _log = Logger()

    def __init__(
        self,
        filename,
        delim=b":",
        usernameField=0,
        passwordField=1,
        caseSensitive=True,
        hash=None,
        cache=False,
    ):
        """
        @type filename: L{str}
        @param filename: The name of the file from which to read username and
        password information.

        @type delim: L{bytes}
        @param delim: The field delimiter used in the file.

        @type usernameField: L{int}
        @param usernameField: The index of the username after splitting a
        line on the delimiter.

        @type passwordField: L{int}
        @param passwordField: The index of the password after splitting a
        line on the delimiter.

        @type caseSensitive: L{bool}
        @param caseSensitive: If true, consider the case of the username when
        performing a lookup.  Ignore it otherwise.

        @type hash: Three-argument callable or L{None}
        @param hash: A function used to transform the plaintext password
        received over the network to a format suitable for comparison
        against the version stored on disk.  The arguments to the callable
        are the username, the network-supplied password, and the in-file
        version of the password.  If the return value compares equal to the
        version stored on disk, the credentials are accepted.

        @type cache: L{bool}
        @param cache: If true, maintain an in-memory cache of the
        contents of the password file.  On lookups, the mtime of the
        file will be checked, and the file will only be re-parsed if
        the mtime is newer than when the cache was generated.
        """
        self.filename = filename
        self.delim = delim
        self.ufield = usernameField
        self.pfield = passwordField
        self.caseSensitive = caseSensitive
        self.hash = hash
        self.cache = cache

        if self.hash is None:
            # The passwords are stored plaintext.  We can support both
            # plaintext and hashed passwords received over the network.
            self.credentialInterfaces = (
                credentials.IUsernamePassword,
                credentials.IUsernameHashedPassword,
            )
        else:
            # The passwords are hashed on disk.  We can support only
            # plaintext passwords received over the network.
            self.credentialInterfaces = (credentials.IUsernamePassword,)

    def __getstate__(self):
        d = dict(vars(self))
        for k in "_credCache", "_cacheTimestamp":
            try:
                del d[k]
            except KeyError:
                pass
        return d

    def _cbPasswordMatch(self, matched, username):
        if matched:
            return username
        else:
            return failure.Failure(error.UnauthorizedLogin())

    def _loadCredentials(self):
        """
        Loads the credentials from the configured file.

        @return: An iterable of C{username, password} couples.
        @rtype: C{iterable}

        @raise UnauthorizedLogin: when failing to read the credentials from the
            file.
        """
        try:
            with open(self.filename, "rb") as f:
                for line in f:
                    line = line.rstrip()
                    parts = line.split(self.delim)

                    if self.ufield >= len(parts) or self.pfield >= len(parts):
                        continue
                    if self.caseSensitive:
                        yield parts[self.ufield], parts[self.pfield]
                    else:
                        yield parts[self.ufield].lower(), parts[self.pfield]
        except IOError as e:
            self._log.error("Unable to load credentials db: {e!r}", e=e)
            raise error.UnauthorizedLogin()

    def getUser(self, username):
        """
        Look up the credentials for a username.

        @param username: The username to look up.
        @type username: L{bytes}

        @returns: Two-tuple of the canonicalicalized username (i.e. lowercase
        if the database is not case sensitive) and the associated password
        value, both L{bytes}.
        @rtype: L{tuple}

        @raises KeyError: When lookup of the username fails.
        """
        if not self.caseSensitive:
            username = username.lower()

        if self.cache:
            if (
                self._credCache is None
                or os.path.getmtime(self.filename) > self._cacheTimestamp
            ):
                self._cacheTimestamp = os.path.getmtime(self.filename)
                self._credCache = dict(self._loadCredentials())
            return username, self._credCache[username]
        else:
            for u, p in self._loadCredentials():
                if u == username:
                    return u, p
            raise KeyError(username)

    def requestAvatarId(self, c):
        try:
            u, p = self.getUser(c.username)
        except KeyError:
            return defer.fail(error.UnauthorizedLogin())
        else:
            up = credentials.IUsernamePassword(c, None)
            if self.hash:
                if up is not None:
                    h = self.hash(up.username, up.password, p)
                    if h == p:
                        return defer.succeed(u)
                return defer.fail(error.UnauthorizedLogin())
            else:
                return defer.maybeDeferred(c.checkPassword, p).addCallback(
                    self._cbPasswordMatch, u
                )


# For backwards compatibility
# Allow access as the old name.
OnDiskUsernamePasswordDatabase = FilePasswordDB
