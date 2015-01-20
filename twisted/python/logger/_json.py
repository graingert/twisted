# -*- test-case-name: twisted.python.logger.test.test_json -*-
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

"""
Tools for saving and loading log events in a structured format.
"""

import types
from json import dumps, loads
from uuid import UUID

from ._flatten import flattenEvent
from ._file import FileLogObserver
from ._levels import LogLevel
from ._logger import Logger
from twisted.python.constants import NamedConstant

from twisted.python.compat import unicode
from twisted.python.failure import Failure

log = Logger()



def failureAsJSON(failure):
    """
    Convert a failure to a JSON-serializable data structure.

    @param failure: A failure to serialize.
    @type failure: L{Failure}

    @return: a mapping of strings to ... stuff, mostly reminiscent of
        L{Failure.__getstate__}
    @rtype: L{dict}
    """
    return dict(
        failure.__getstate__(),
        type=dict(
            __module__=failure.type.__module__,
            __name__=failure.type.__name__,
        )
    )



def asBytes(obj):
    """
    On Python 2, we really need native strings in a variety of places;
    attribute names will sort of work in a __dict__, but they're subtly wrong;
    however, printing tracebacks relies on I/O to containers that only support
    bytes.  This function converts _all_ native strings within a
    JSON-deserialized object to bytes.

    @param obj: A object to convert to bytes.
    @type obj: L{object}

    @return: A string of UTF-8 bytes.
    @rtype: L{bytes}
    """
    if isinstance(obj, list):
        return map(asBytes, obj)
    elif isinstance(obj, dict):
        return dict((asBytes(k), asBytes(v)) for k, v in obj.items())
    elif isinstance(obj, unicode):
        return obj.encode("utf-8")
    else:
        return obj



def failureFromJSON(failureDict):
    """
    Load a L{Failure} from a dictionary deserialized from JSON.

    @param failureDict: a JSON-deserialized object like one previously returned
        by L{failureAsJSON}.
    @type failureDict: L{dict} mapping L{unicode} to attributes

    @return: L{Failure}
    @rtype: L{Failure}
    """
    newFailure = getattr(Failure, "__new__", None)
    if newFailure is None:
        failureDict = asBytes(failureDict)
        f = types.InstanceType(Failure)
    else:
        f = newFailure(Failure)
    typeInfo = failureDict["type"]
    failureDict["type"] = type(typeInfo["__name__"], (), typeInfo)
    f.__dict__ = failureDict
    return f



classInfo = [
    (
        lambda level: (
            isinstance(level, NamedConstant) and
            getattr(LogLevel, level.name, None) is level
        ),
        UUID("02E59486-F24D-46AD-8224-3ACDF2A5732A"),
        lambda level: dict(name=level.name),
        lambda level: getattr(LogLevel, level["name"], None)
    ),

    (
        lambda o: isinstance(o, Failure),
        UUID("E76887E2-20ED-49BF-A8F8-BA25CC586F2D"),
        failureAsJSON, failureFromJSON
    ),
]



uuidToLoader = dict([
    (uuid, loader) for (predicate, uuid, saver, loader) in classInfo
])



def objectLoadHook(aDict):
    """
    Dictionary-to-object-translation hook for certain value types used within
    the logging system.

    @see: the C{object_hook} parameter to L{json.load}

    @param aDict: A dictionary loaded from a JSON object.
    @type aDict: L{dict}

    @return: C{aDict} itself, or the object represented by C{aDict}
    @rtype: L{object}
    """
    if "__class_uuid__" in aDict:
        return uuidToLoader[UUID(aDict["__class_uuid__"])](aDict)
    return aDict



def objectSaveHook(pythonObject):
    """
    Object-to-serializable hook for certain value types used within the logging
    system.

    @see: the C{default} parameter to L{json.dump}

    @param pythonObject: Any object.
    @type pythonObject: L{object}

    @return: If the object is one of the special types the logging system
        supports, a specially-formatted dictionary; otherwise, a marker
        dictionary indicating that it could not be serialized.
    """
    for (predicate, uuid, saver, loader) in classInfo:
        if predicate(pythonObject):
            result = saver(pythonObject)
            result["__class_uuid__"] = str(uuid)
            return result
    return {"unpersistable": True}



def eventAsJSON(event):
    """
    Encode an event as JSON, flattening it if necessary to preserve as much
    structure as possible.

    Not all structure from the log event will be preserved when it is
    serialized.

    @param event: A log event dictionary.
    @type event: L{dict} with arbitrary keys and values

    @return: A string of the serialized JSON; note that this will contain no
        newline characters, and may thus safely be stored in a line-delimited
        file.
    @rtype: L{unicode}
    """
    if bytes is str:
        kw = dict(default=objectSaveHook, encoding="charmap", skipkeys=True)
    else:
        def default(unencodable):
            """
            Serialize an object not otherwise serializable by L{dumps}.

            @param unencodable: An unencodable object.
            @return: C{unencodable}, serialized
            """
            if isinstance(unencodable, bytes):
                return unencodable.decode("charmap")
            return objectSaveHook(unencodable)

        kw = dict(default=default, skipkeys=True)

    flattenEvent(event)
    result = dumps(event, **kw)
    if not isinstance(result, unicode):
        return unicode(result, "utf-8", "replace")
    return result



def eventFromJSON(eventText):
    """
    Decode a log event from JSON.

    @param eventText: The output of a previous call to L{eventAsJSON}
    @type eventText: L{unicode}

    @return: A reconstructed version of the log event.
    @rtype: L{dict}
    """
    loaded = loads(eventText, object_hook=objectLoadHook)
    return loaded



def jsonFileLogObserver(outFile):
    """
    Create a L{FileLogObserver} that emits JSON-serialized events to a
    specified (writable) file-like object.

    Events are written according to the IETF draft document
    "draft-ietf-json-text-sequence-13", meaning that JSON text
    is encoded as UTF-8 bytes, preceded by a record separator character
    (C{u"\x1e"}) and followed by a line feed character (C{u"\n"}).

    @param outFile: A file-like object.  Ideally one should be passed which
        accepts L{unicode} data.  Otherwise, UTF-8 L{bytes} will be used.
    @type outFile: L{io.IOBase}

    @return: A file log observer.
    @rtype: L{FileLogObserver}
    """
    return FileLogObserver(
        outFile,
        lambda event: u"\x1e{0}\n".format(eventAsJSON(event))
    )



def eventsFromJSONLogFile(inFile):
    """
    Load events from a file previously saved with L{jsonFileLogObserver}.

    @param inFile: A (readable) file-like object.  Data read from C{inFile}
        should be L{unicode} or UTF-8 L{bytes}.
    @type inFile: iterable of lines

    @return: Log events as read from C{inFile}.
    @rtype: iterable of L{dict}
    """
    def eventFromRecord(record):
        if record and record[-1] == 10:  # 10 is "\n"
            try:
                return eventFromJSON(bytes(record))
            except ValueError:
                log.error(
                    u"Unable to read JSON record: {record}", record=record
                )

    buffer = bytearray()

    while True:
        newData = inFile.read(4096)

        if not newData:
            event = eventFromRecord(buffer)
            if event is not None:
                yield event
            break

        buffer += newData.encode("utf-8")
        records = buffer.split(b"\x1e")

        for record in records[:-1]:
            event = eventFromRecord(record)
            if event is not None:
                yield event

        buffer = records[-1]