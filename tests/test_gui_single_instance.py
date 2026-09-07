from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtWidgets import QApplication
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

from vertex_proxy.gui import (
    activate_existing_instance,
    setup_single_instance_server,
    VertexProxyApp,
)


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_activate_existing_instance_when_no_server(qapp):
    test_key = "test_single_instance_none_key"
    QLocalServer.removeServer(test_key)
    # When no server is listening, should return False
    assert activate_existing_instance(test_key) is False


def test_activate_existing_instance_wakes_window(qapp):
    test_key = "test_single_instance_active_key"
    QLocalServer.removeServer(test_key)

    mock_window = MagicMock()
    server = setup_single_instance_server(mock_window, key=test_key)

    try:
        # Now attempting to activate should succeed (return True)
        result = activate_existing_instance(test_key)
        assert result is True

        # Process pending Qt events so signals are dispatched
        qapp.processEvents()

        # Check that window.bring_to_front was called
        mock_window.bring_to_front.assert_called()
    finally:
        server.close()
        QLocalServer.removeServer(test_key)


def test_bring_to_front_method(qapp):
    with patch.object(VertexProxyApp, "load_settings"), \
         patch.object(VertexProxyApp, "update_adc_status"), \
         patch.object(VertexProxyApp, "setup_tray"):
        window = VertexProxyApp()
        
        # Test calling bring_to_front does not throw and calls show / activate
        with patch.object(window, "show") as mock_show, \
             patch.object(window, "raise_") as mock_raise, \
             patch.object(window, "activateWindow") as mock_activate:
            window.bring_to_front()
            mock_show.assert_called_once()
            mock_raise.assert_called_once()
            mock_activate.assert_called_once()
