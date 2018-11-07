// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-16
 *
 */
//----------------------------------------------------------------------
#ifndef ULS_UDP_LOGGING_SERVER_H_INCLUDED
#define ULS_UDP_LOGGING_SERVER_H_INCLUDED

#include <QObject>
#include <QUdpSocket>
#include <QSqlDatabase>

class UdpLoggingServer : QObject
{
  Q_OBJECT

  public:
  UdpLoggingServer(const QString& db_filename);
  virtual ~UdpLoggingServer();

private slots:
  void processPendingDatagrams();

private:
  QUdpSocket *m_udp_socket;
  QSqlDatabase m_db;
};

#endif
