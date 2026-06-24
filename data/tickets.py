"""
Sample IT support ticket dataset for the IT Support RAG Agent.

Each ticket is a dictionary with the following fields:
    - ticket_id: str            e.g. "TKT-001"
    - category: str             one of the 8 supported categories
    - issue: str                what the employee reported
    - resolution: str           how IT resolved it (2-3 sentences)
    - resolved_in_minutes: int  realistic time-to-resolution (5-120)

Categories covered (>= 5 tickets each):
    VPN, Password, Software Access, Hardware,
    Email, Printer, Network, ERP Access

Data is fictional and modeled on a large manufacturing company ("Enterprise IT Support").
"""

TICKETS = [
    # ----------------------------------------------------------------- VPN
    {
        "ticket_id": "TKT-001",
        "category": "VPN",
        "issue": "Remote engineer in the Grand Rapids plant cannot connect to "
                 "Cisco AnyConnect; client returns 'Login failed' after entering "
                 "credentials.",
        "resolution": "Found the user's Active Directory account was locked due to "
                       "repeated failed MFA prompts. Unlocked the account in AD, reset "
                       "the Duo MFA device pairing, and confirmed AnyConnect connected "
                       "successfully to the GR-VPN gateway.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-002",
        "category": "VPN",
        "issue": "Sales rep reports Cisco AnyConnect connects but cannot reach the "
                 "internal SAP server or shared drives while on hotel Wi-Fi.",
        "resolution": "Identified a split-tunnel routing conflict caused by the hotel's "
                       "192.168.1.0/24 subnet overlapping with an internal range. Pushed "
                       "the full-tunnel VPN profile to the user and had them reconnect, "
                       "which restored access to SAP and the file shares.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-003",
        "category": "VPN",
        "issue": "New hire in Finance received the error 'The VPN client was unable to "
                 "establish a connection' the first time launching AnyConnect.",
        "resolution": "The AnyConnect 4.10 client predated the upgraded ASA gateway and "
                       "failed the version compatibility check. Pushed the current 5.1 "
                       "client via SCCM, rebooted the laptop, and verified a clean tunnel "
                       "to the corporate gateway.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-004",
        "category": "VPN",
        "issue": "Plant supervisor's AnyConnect session drops every 10-15 minutes when "
                 "working from home, forcing repeated reconnections.",
        "resolution": "Traced the drops to an aggressive idle-timeout on the user's "
                       "consumer router plus a low DPD value in the group policy. Raised "
                       "the dead-peer-detection interval on the VPN profile and advised "
                       "disabling the router's Wi-Fi power-saving mode, which stabilized "
                       "the session.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-005",
        "category": "VPN",
        "issue": "Contractor cannot install Cisco AnyConnect; Windows blocks it with "
                 "'This app has been blocked by your system administrator.'",
        "resolution": "An AppLocker policy was blocking the unsigned contractor install "
                       "package. Provided the IT-signed MSI from the software portal and "
                       "granted a temporary AppLocker exception, after which the install "
                       "completed and the contractor connected to the partner VPN.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-006",
        "category": "VPN",
        "issue": "User on a new ISP gets 'Connection attempt has timed out' when "
                 "launching AnyConnect, though web browsing works fine.",
        "resolution": "The new ISP was blocking outbound UDP 443 used by DTLS. Switched "
                       "the user's AnyConnect profile to fall back to TLS over TCP 443 "
                       "and confirmed the tunnel established normally on the next attempt.",
        "resolved_in_minutes": 45,
    },

    # ------------------------------------------------------------ Password
    {
        "ticket_id": "TKT-007",
        "category": "Password",
        "issue": "Warehouse employee is locked out of their Windows account after "
                 "returning from a two-week vacation and forgetting the password.",
        "resolution": "Verified identity through the manager and employee ID, then reset "
                       "the Active Directory password and cleared the account lockout. "
                       "Walked the user through setting a new password at next logon and "
                       "confirmed they could sign in.",
        "resolved_in_minutes": 10,
    },
    {
        "ticket_id": "TKT-008",
        "category": "Password",
        "issue": "User's Microsoft 365 password expired and they cannot reset it because "
                 "their registered phone number for MFA is out of date.",
        "resolution": "Confirmed identity using a video call and badge photo, then reset "
                       "the password in the Microsoft 365 admin center and removed the "
                       "stale MFA method. Helped the user re-register the Microsoft "
                       "Authenticator app and complete a successful sign-in.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-009",
        "category": "Password",
        "issue": "Engineer reports the self-service password reset portal rejects every "
                 "new password with 'does not meet complexity requirements.'",
        "resolution": "The user was reusing a password from their history and falling "
                       "below the 14-character minimum. Explained the complexity and "
                       "history policy, suggested a passphrase format, and confirmed the "
                       "new password was accepted in the SSPR portal.",
        "resolved_in_minutes": 15,
    },
    {
        "ticket_id": "TKT-010",
        "category": "Password",
        "issue": "Shift worker shares a kiosk PC and is repeatedly locked out because "
                 "an old mapped session keeps submitting their previous password.",
        "resolution": "Found a stale credential cached in Windows Credential Manager and "
                       "an old Outlook profile retrying the previous password. Cleared "
                       "the cached credentials, reset the AD password, and removed the "
                       "lockout, which stopped the recurring lockouts.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-011",
        "category": "Password",
        "issue": "Manager cannot log in to ServiceNow; password works for email but "
                 "ServiceNow shows 'Invalid username or password.'",
        "resolution": "ServiceNow was using a cached local credential that had not synced "
                       "after the user's last AD password change. Forced a SSO/SAML "
                       "re-authentication and cleared the browser's saved ServiceNow "
                       "credentials, after which the user logged in successfully.",
        "resolved_in_minutes": 18,
    },
    {
        "ticket_id": "TKT-012",
        "category": "Password",
        "issue": "Production planner forgot their SAP password and used all remaining "
                 "logon attempts, locking the SAP user ID.",
        "resolution": "Confirmed the user's identity and reset the SAP password in "
                       "transaction SU01, setting it to require a change at next logon. "
                       "Reset the failed logon counter and verified the planner could "
                       "access the production module.",
        "resolved_in_minutes": 22,
    },

    # ------------------------------------------------------- Software Access
    {
        "ticket_id": "TKT-013",
        "category": "Software Access",
        "issue": "New designer needs access to AutoCAD but the application reports "
                 "'No license available' when launching.",
        "resolution": "Confirmed the designer was not yet in the AutoCAD licensing "
                       "security group on the Autodesk network license server. Added the "
                       "user to the CAD-Users group, refreshed group membership, and "
                       "verified AutoCAD checked out a license successfully.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-014",
        "category": "Software Access",
        "issue": "Finance analyst requests access to Microsoft Power BI but cannot open "
                 "the shared workspace, seeing 'You don't have permission.'",
        "resolution": "The analyst had a Power BI Free license but the workspace required "
                       "Pro. Assigned a Power BI Pro license in the Microsoft 365 admin "
                       "center and added the user to the Finance workspace as a Viewer, "
                       "confirming the reports loaded.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-015",
        "category": "Software Access",
        "issue": "Employee installed Microsoft Teams but it crashes on launch with a "
                 "white screen immediately after the splash logo.",
        "resolution": "A corrupted Teams cache was causing the crash. Fully exited Teams, "
                       "cleared the %appdata%\\Microsoft\\Teams cache folders, and "
                       "reinstalled the new Teams client, after which it launched and "
                       "signed in normally.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-016",
        "category": "Software Access",
        "issue": "Quality engineer needs the Minitab statistical package installed on a "
                 "new workstation for an upcoming audit.",
        "resolution": "Submitted and approved the software request in ServiceNow, then "
                       "deployed Minitab through SCCM to the engineer's machine. Activated "
                       "the multi-user license against the license server and confirmed "
                       "the application opened with full functionality.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-017",
        "category": "Software Access",
        "issue": "User cannot open shared Excel files from SharePoint; Microsoft 365 "
                 "prompts to activate and says 'Product Deactivated.'",
        "resolution": "The Microsoft 365 Apps license had been unassigned during a recent "
                       "license cleanup. Reassigned an E3 license to the user, signed out "
                       "and back into Office, and confirmed activation restored editing of "
                       "the SharePoint files.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-018",
        "category": "Software Access",
        "issue": "Procurement specialist needs edit access to a restricted Microsoft "
                 "SharePoint site but currently only has read-only permission.",
        "resolution": "Verified manager approval for the elevated access in ServiceNow, "
                       "then added the specialist to the SharePoint 'Members' group with "
                       "Edit rights. Confirmed with the user that they could now upload "
                       "and modify documents in the procurement library.",
        "resolved_in_minutes": 20,
    },

    # ------------------------------------------------------------- Hardware
    {
        "ticket_id": "TKT-019",
        "category": "Hardware",
        "issue": "Plant manager's Dell Latitude laptop will not power on; no lights and "
                 "no response when plugged into the dock.",
        "resolution": "Isolated the fault to a failed 90W dock power adapter, not the "
                       "laptop. Connected the laptop directly to a known-good charger, "
                       "performed a battery-drain reset, and replaced the faulty dock "
                       "adapter, after which the laptop booted normally.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-020",
        "category": "Hardware",
        "issue": "Engineer's external monitor shows 'No Signal' when connected to the "
                 "Dell WD19 dock via USB-C, though the laptop screen works.",
        "resolution": "Updated the WD19 dock firmware and the Intel/DisplayLink graphics "
                       "drivers, which were out of date after a Windows feature update. "
                       "Reseated the USB-C cable and confirmed both external monitors were "
                       "detected and extended correctly.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-021",
        "category": "Hardware",
        "issue": "Front-desk receptionist's keyboard and mouse stopped responding after "
                 "a power outage at the Kentwood site.",
        "resolution": "The USB hub the peripherals were connected to had not re-enumerated "
                       "after the outage. Power-cycled the workstation, moved the keyboard "
                       "and mouse to direct USB ports, and verified both devices worked; "
                       "replaced the failing USB hub from stock.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-022",
        "category": "Hardware",
        "issue": "User reports their laptop is extremely slow and the fan runs constantly, "
                 "making it nearly unusable during the day.",
        "resolution": "Diagnostics showed the SSD was 98% full and thermal throttling from "
                       "dust-clogged vents. Cleared temp files and offloaded data to "
                       "OneDrive to free space, physically cleaned the vents, and confirmed "
                       "CPU temperatures and performance returned to normal.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-023",
        "category": "Hardware",
        "issue": "Field technician needs a replacement laptop battery; the existing one "
                 "lasts only 20 minutes and shows 'Plugged in, not charging.'",
        "resolution": "Battery diagnostics reported the cell at 41% design capacity and "
                       "flagged for replacement. Ordered and installed a new OEM battery, "
                       "calibrated it through a full charge cycle, and confirmed runtime "
                       "returned to roughly four hours.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-024",
        "category": "Hardware",
        "issue": "Conference room webcam and microphone are not detected by the room PC "
                 "during Microsoft Teams meetings.",
        "resolution": "The Logitech Rally USB cable had partially disconnected behind the "
                       "credenza and the device firmware needed an update. Reseated the "
                       "cable, updated firmware via Logi Tune, and verified the camera and "
                       "mic appeared as selectable devices in Teams.",
        "resolved_in_minutes": 30,
    },

    # ---------------------------------------------------------------- Email
    {
        "ticket_id": "TKT-025",
        "category": "Email",
        "issue": "User's Outlook is stuck on 'Trying to connect...' and no new mail has "
                 "arrived since this morning.",
        "resolution": "Found a corrupted Outlook OST profile after a recent mailbox move. "
                       "Recreated the Outlook profile pointing at the new Exchange Online "
                       "endpoint and let it resync, which restored the connection and "
                       "delivered the queued mail.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-026",
        "category": "Email",
        "issue": "Manager reports legitimate emails from a key supplier are landing in "
                 "the Junk folder in Outlook every time.",
        "resolution": "Reviewed the message headers and confirmed a soft spam score from "
                       "Microsoft Defender. Added the supplier domain to the tenant "
                       "allow-list in the Exchange admin center and to the user's Safe "
                       "Senders, after which new supplier mail arrived in the Inbox.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-027",
        "category": "Email",
        "issue": "Employee accidentally deleted an important email thread and emptied the "
                 "Deleted Items folder in Outlook.",
        "resolution": "Used the Recover Deleted Items feature in Outlook to restore the "
                       "thread from the dumpster, and confirmed with the Exchange Online "
                       "retention policy that the items were still recoverable. Restored "
                       "the messages back to the user's Inbox.",
        "resolved_in_minutes": 15,
    },
    {
        "ticket_id": "TKT-028",
        "category": "Email",
        "issue": "User's mailbox is full and Outlook is rejecting outgoing mail with "
                 "'Your mailbox is over its size limit.'",
        "resolution": "Confirmed the mailbox had hit its 100 GB quota largely from old "
                       "attachments. Enabled the Online Archive in Exchange Online and "
                       "moved items older than two years into it, bringing the primary "
                       "mailbox under quota so mail flowed again.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-029",
        "category": "Email",
        "issue": "Shared department mailbox is no longer visible in Outlook for two team "
                 "members after a recent reorg.",
        "resolution": "The two users had been removed from the mailbox's Full Access "
                       "permission group during the reorg. Re-granted Full Access and "
                       "Send As on the shared mailbox in the Exchange admin center, then "
                       "had Outlook restarted so the mailbox auto-mapped back in.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-030",
        "category": "Email",
        "issue": "Executive assistant cannot send email on behalf of the VP; Outlook "
                 "returns 'You do not have permission to send as this user.'",
        "resolution": "Verified the request and added 'Send on Behalf' delegate rights "
                       "for the assistant on the VP's mailbox in Exchange Online. Had the "
                       "assistant restart Outlook and confirmed a test message sent on "
                       "behalf of the VP delivered correctly.",
        "resolved_in_minutes": 20,
    },

    # -------------------------------------------------------------- Printer
    {
        "ticket_id": "TKT-031",
        "category": "Printer",
        "issue": "Accounting team cannot print to the shared HP LaserJet on the third "
                 "floor; jobs sit in the queue with status 'Error.'",
        "resolution": "The print spooler on the print server had hung, stalling the queue. "
                       "Cleared the stuck jobs, restarted the Print Spooler service, and "
                       "confirmed a test page printed; updated the HP Universal Print "
                       "Driver to prevent recurrence.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-032",
        "category": "Printer",
        "issue": "User's documents print with faded text and vertical streaks on the "
                 "shop-floor multifunction printer.",
        "resolution": "Identified a low toner cartridge and a dirty drum unit as the cause "
                       "of the streaking. Replaced the toner cartridge, ran the printer's "
                       "drum-cleaning cycle, and printed a test page confirming clean, "
                       "dark output.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-033",
        "category": "Printer",
        "issue": "New employee cannot find the department printer when trying to add it "
                 "on their laptop.",
        "resolution": "The user was not receiving the printer via Group Policy because "
                       "their machine was in the wrong OU. Moved the computer object to "
                       "the correct site OU, ran a Group Policy update, and confirmed the "
                       "department printer auto-installed and printed a test page.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-034",
        "category": "Printer",
        "issue": "Badge-release printing fails for a manager; the printer says "
                 "'Authentication failed' when they tap their badge.",
        "resolution": "The user's badge ID was not linked to their account in the secure "
                       "print system after a badge reissue. Re-associated the new badge "
                       "number with the user in the print management console and confirmed "
                       "a held job released successfully on the next tap.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-035",
        "category": "Printer",
        "issue": "Large-format plotter in engineering jams on every print and shows a "
                 "'Paper Jam in Roll 2' error.",
        "resolution": "Found a misaligned roll and torn media triggering the jam sensor. "
                       "Cleared the torn paper, reloaded and squared Roll 2, and ran the "
                       "media-advance calibration, after which several test plots printed "
                       "without jamming.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-036",
        "category": "Printer",
        "issue": "Office printer is showing offline for everyone on the floor even though "
                 "it is powered on and shows ready.",
        "resolution": "The printer had picked up a new DHCP address while its print queue "
                       "still pointed at the old IP. Set a DHCP reservation for the "
                       "printer's MAC, updated the port on the print server to the new "
                       "address, and confirmed the queue came back online for all users.",
        "resolved_in_minutes": 35,
    },

    # -------------------------------------------------------------- Network
    {
        "ticket_id": "TKT-037",
        "category": "Network",
        "issue": "Entire row of cubicles in the engineering wing lost wired network "
                 "connectivity simultaneously this morning.",
        "resolution": "Traced the outage to a failed access switch in the wiring closet "
                       "serving that row. Swapped in a spare switch from inventory, "
                       "restored the saved configuration, and confirmed all affected "
                       "ports came back online with full connectivity.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-038",
        "category": "Network",
        "issue": "User in the warehouse reports very slow file transfers and intermittent "
                 "drops on the corporate Wi-Fi near the loading dock.",
        "resolution": "A site survey showed weak coverage and channel interference near "
                       "the dock. Added a new access point on a clean 5 GHz channel and "
                       "adjusted the existing AP's power, which raised the signal level "
                       "and eliminated the drops for the user.",
        "resolved_in_minutes": 75,
    },
    {
        "ticket_id": "TKT-039",
        "category": "Network",
        "issue": "Employee's laptop connects to Wi-Fi but shows 'No Internet, secured' "
                 "and cannot reach any internal or external sites.",
        "resolution": "The laptop had pulled an APIPA 169.254 address because the VLAN's "
                       "DHCP scope was exhausted. Expanded the DHCP scope on the server, "
                       "released and renewed the client's address, and confirmed full "
                       "internal and internet access.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-040",
        "category": "Network",
        "issue": "Several users in Finance cannot reach an internal web application, "
                 "getting 'This site can't be reached' while other sites work.",
        "resolution": "Found a stale DNS record pointing the app's hostname at a "
                       "decommissioned server. Updated the internal DNS A record to the "
                       "new server IP and flushed DNS on the affected clients, restoring "
                       "access to the application.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-041",
        "category": "Network",
        "issue": "Guest contractors cannot get online through the guest Wi-Fi; the "
                 "captive portal page never loads after they connect.",
        "resolution": "The guest network's captive-portal service had stopped on the "
                       "wireless controller after a maintenance reboot. Restarted the "
                       "captive-portal service and validated the splash page loaded, then "
                       "had a contractor confirm they could authenticate and browse.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-042",
        "category": "Network",
        "issue": "User reports frequent brief disconnects on their wired connection that "
                 "interrupt Microsoft Teams calls several times an hour.",
        "resolution": "Switch logs showed the port flapping due to a damaged patch cable. "
                       "Replaced the patch cable between the wall jack and the laptop dock, "
                       "confirmed the port stayed up with no errors, and verified a Teams "
                       "call held steadily afterward.",
        "resolved_in_minutes": 30,
    },

    # ------------------------------------------------------------ ERP Access
    {
        "ticket_id": "TKT-043",
        "category": "ERP Access",
        "issue": "New buyer cannot open the purchasing module in SAP; the system returns "
                 "'No authorization for transaction ME21N.'",
        "resolution": "Confirmed the buyer's SAP role did not include purchasing "
                       "authorizations. Submitted a role request, assigned the "
                       "Z_PURCHASING composite role in SU01, and verified the user could "
                       "create a purchase order in ME21N.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-044",
        "category": "ERP Access",
        "issue": "Production scheduler reports SAP is extremely slow and times out when "
                 "running the MRP planning report.",
        "resolution": "Found the user running an unrestricted MRP report pulling months "
                       "of data. Coached the scheduler on adding plant and date filters "
                       "and scheduled the heavy run as a background job in SM37, which "
                       "returned results without timing out.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-045",
        "category": "ERP Access",
        "issue": "Finance user's SAP GUI shows 'Partner not reached' and will not "
                 "connect to the production system this morning.",
        "resolution": "The SAP message server entry in the user's SAP GUI was pointing at "
                       "a retired application server. Updated the connection in SAP Logon "
                       "to the load-balanced group entry and confirmed the user logged "
                       "into the production system successfully.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-046",
        "category": "ERP Access",
        "issue": "Warehouse clerk cannot post goods receipts in SAP; transaction MIGO "
                 "errors with 'You are not authorized for movement type 101.'",
        "resolution": "The clerk's inventory role was missing the movement-type "
                       "authorization object for receipts. Added the required "
                       "authorization to their inventory role, refreshed the user buffer, "
                       "and verified a successful goods receipt posting in MIGO.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-047",
        "category": "ERP Access",
        "issue": "Manager requests SAP access for a transferred employee who needs the "
                 "same authorizations as their previous team.",
        "resolution": "Reviewed the source role and confirmed segregation-of-duties had "
                       "no conflicts in GRC. Copied the appropriate display and posting "
                       "roles to the employee's SAP ID, and confirmed with the user that "
                       "they could open the required transactions for the new team.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-048",
        "category": "ERP Access",
        "issue": "User's SAP account was locked after the system flagged it as dormant "
                 "following an extended leave.",
        "resolution": "Confirmed the account was auto-locked by the dormancy policy after "
                       "90 days of inactivity. Reactivated the SAP user in SU01, reset the "
                       "validity dates, and had the user log in and change their password "
                       "to confirm access was restored.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-049",
        "category": "ERP Access",
        "issue": "Plant controller cannot run a custom cost-center report in SAP; it "
                 "ends with 'Runtime error TIME_OUT' before finishing.",
        "resolution": "The report was selecting too wide a date range against an "
                       "unindexed field. Worked with the controller to narrow the "
                       "selection to a single fiscal period and ran the report as a "
                       "background job, which completed and produced the spool output.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-050",
        "category": "ERP Access",
        "issue": "Quality manager needs read-only access to the SAP QM module to review "
                 "inspection lots but currently has no QM authorization.",
        "resolution": "Confirmed manager approval in ServiceNow and assigned the "
                       "Z_QM_DISPLAY read-only role to the user's SAP account. Verified "
                       "the quality manager could open and view inspection lots in QA32 "
                       "without any edit capability.",
        "resolved_in_minutes": 35,
    },

    # ------------------------------------------------------------- VPN (cont.)
    {
        "ticket_id": "TKT-051",
        "category": "VPN",
        "issue": "New remote hire's first-day VPN access is not working; AnyConnect "
                 "installs but the account shows 'not provisioned for remote access.'",
        "resolution": "The onboarding workflow had created the AD account but the "
                       "automated remote-access group assignment failed silently. "
                       "Manually added the user to the VPN-Users security group, pushed "
                       "the AnyConnect profile, and walked the new hire through their "
                       "first successful connection on a welcome call.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-052",
        "category": "VPN",
        "issue": "After an AnyConnect auto-update overnight, several engineers lose "
                 "access to internal file shares and SAP while VPN still shows connected.",
        "resolution": "The update had silently reset custom split-tunnel routes back to "
                       "vendor defaults. Reapplied the corporate split-tunnel policy "
                       "through the ASA group policy, pushed it to affected clients, and "
                       "disabled client-side auto-update pending a controlled rollout.",
        "resolved_in_minutes": 70,
    },
    {
        "ticket_id": "TKT-053",
        "category": "VPN",
        "issue": "User's VPN tunnel is active but SSO tokens for SAP and SharePoint keep "
                 "expiring every few minutes, forcing repeated re-logins on both systems.",
        "resolution": "Found the VPN gateway's clock had drifted out of sync with the "
                       "identity provider, invalidating SAML tokens almost immediately. "
                       "Corrected the NTP source on the gateway, forced a resync, and "
                       "confirmed both SAP and SharePoint sessions held normally afterward.",
        "resolved_in_minutes": 65,
    },
    {
        "ticket_id": "TKT-054",
        "category": "VPN",
        "issue": "Engineer's laptop repeatedly fails certificate validation when "
                 "connecting to AnyConnect; Tier 1 reset the profile twice with no change.",
        "resolution": "Escalated to the Tier 2 network team after standard profile resets "
                       "failed. They identified an expired intermediate CA certificate in "
                       "the device's trust store, pushed the updated certificate chain via "
                       "Intune, and confirmed the connection validated successfully.",
        "resolved_in_minutes": 95,
    },
    {
        "ticket_id": "TKT-055",
        "category": "VPN",
        "issue": "User can reach the VPN gateway but every internal hostname fails to "
                 "resolve, while raw IP addresses work fine over the tunnel.",
        "resolution": "A split-DNS misconfiguration was sending internal queries to the "
                       "public resolver instead of the internal DNS servers. Corrected the "
                       "DNS suffix and server list in the AnyConnect profile, flushed the "
                       "client's DNS cache, and confirmed hostnames resolved correctly.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-056",
        "category": "VPN",
        "issue": "Contractor's VPN profile was never provisioned because their visa "
                 "paperwork delayed final HR approval, and their start date is tomorrow.",
        "resolution": "Coordinated with HR and the hiring manager to get an emergency "
                       "provisional approval logged in ServiceNow, then provisioned a "
                       "time-limited contractor VPN profile scoped to the partner network "
                       "only, ready before the contractor's first day.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-057",
        "category": "VPN",
        "issue": "After an Intune policy update enabling Always On VPN, several laptops "
                 "now have two conflicting VPN profiles and connections fail intermittently.",
        "resolution": "The new Always On profile had been pushed alongside the legacy "
                       "manual profile instead of replacing it. Removed the legacy profile "
                       "from affected devices via Intune, redeployed the Always On profile "
                       "cleanly, and verified stable automatic connections on reboot.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-058",
        "category": "VPN",
        "issue": "User's AnyConnect tunnel connects normally, but their Citrix virtual "
                 "desktop session hangs on 'Loading apps' every time they sign in remotely.",
        "resolution": "Traced the issue to a port-blocking conflict between the VPN's "
                       "full-tunnel routing and the Citrix Gateway's ICA traffic. Added a "
                       "routing exception for the Citrix subnet to the split-tunnel ACL "
                       "and confirmed virtual desktop apps loaded normally over VPN.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-059",
        "category": "VPN",
        "issue": "Multiple remote users report intermittent packet loss and dropped "
                 "video calls over VPN throughout the afternoon; Tier 1 found no client issue.",
        "resolution": "Escalated to Tier 2 network engineering, who traced the packet "
                       "loss to a degraded peering link at the ISP serving the VPN "
                       "concentrator. The ISP rerouted traffic during the outage window "
                       "and engineering confirmed loss returned to baseline afterward.",
        "resolved_in_minutes": 110,
    },
    {
        "ticket_id": "TKT-060",
        "category": "VPN",
        "issue": "New hire cannot complete VPN MFA enrollment because Duo Mobile rejects "
                 "their personal phone as 'unsupported device' during setup.",
        "resolution": "The phone's OS version predated Duo's minimum support baseline. "
                       "Issued a temporary hardware token from the IT cage as a stop-gap, "
                       "enrolled it in Duo for the VPN policy, and advised the new hire to "
                       "update their phone OS before re-enrolling the mobile app later.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-061",
        "category": "VPN",
        "issue": "Since a recent Windows update, a user's VPN won't connect at all; the "
                 "client reports 'blocked by local firewall policy.'",
        "resolution": "The update had reclassified the home network profile from Private "
                       "to Public, triggering stricter Windows Firewall rules that blocked "
                       "the VPN service. Reset the network profile to Private, adjusted the "
                       "firewall exception for AnyConnect, and confirmed the tunnel connected.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-062",
        "category": "VPN",
        "issue": "After the recent acquisition, a transferred employee's AnyConnect "
                 "profile keeps connecting to the old company's gateway instead of ours.",
        "resolution": "The legacy profile had not been removed during the account "
                       "migration. Uninstalled the old company's VPN profile, deployed the "
                       "correct corporate gateway profile via SCCM, and confirmed the user "
                       "connected to internal resources on the new network correctly.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-063",
        "category": "VPN",
        "issue": "VPN access is down company-wide; every user gets a certificate trust "
                 "error immediately on connecting.",
        "resolution": "Escalated immediately to the Tier 2 security team, who found the "
                       "VPN gateway's public certificate had expired overnight. They issued "
                       "and installed a renewed certificate from the CA, restarted the VPN "
                       "service, and confirmed connections were restored for all sites.",
        "resolved_in_minutes": 50,
    },

    # -------------------------------------------------------- Password (cont.)
    {
        "ticket_id": "TKT-064",
        "category": "Password",
        "issue": "New hire's temporary password from the HR onboarding portal does not "
                 "work when they try to log in to their laptop on day one.",
        "resolution": "The HR system's password push had not synced to Active Directory "
                       "before the start date due to a delayed nightly sync job. Manually "
                       "set a temporary AD password, forced an immediate sync, and had the "
                       "new hire log in and set a permanent password successfully.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-065",
        "category": "Password",
        "issue": "User reset their Windows password successfully, but SAP still rejects "
                 "the new password as invalid an hour later.",
        "resolution": "Escalated to Tier 2 identity management after the standard SAP "
                       "sync did not pick up the change. They found the AD-to-SAP "
                       "password sync connector had stalled overnight, manually triggered "
                       "a resync for the account, and confirmed SAP accepted the new "
                       "credentials.",
        "resolved_in_minutes": 80,
    },
    {
        "ticket_id": "TKT-066",
        "category": "Password",
        "issue": "Following an SSPR portal update, dozens of users are calling in because "
                 "their usual passwords are now rejected as 'too weak.'",
        "resolution": "The update had silently raised the minimum length and added a "
                       "dictionary-word check without notice. Confirmed the new policy with "
                       "security, published guidance on passphrase formats to the intranet, "
                       "and helped the first wave of callers set compliant passwords.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-067",
        "category": "Password",
        "issue": "Executive's smart card PIN is locked after too many failed attempts and "
                 "they cannot log in to any workstation.",
        "resolution": "Verified identity in person, then walked the executive to the badge "
                       "office to reissue a working smart card. Reset the PIN counter in "
                       "the certificate authority console, re-enrolled the new card to "
                       "their account, and confirmed login succeeded on their workstation.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-068",
        "category": "Password",
        "issue": "A user keeps getting locked out every 30 minutes despite resetting "
                 "their password twice; Tier 1 cannot find the source of the bad attempts.",
        "resolution": "Escalated to Tier 2, who found an old scheduled task on the user's "
                       "previous laptop was still running with cached credentials and "
                       "hammering AD with the old password. Disabled the task remotely, "
                       "reset the password once more, and the lockouts stopped.",
        "resolved_in_minutes": 75,
    },
    {
        "ticket_id": "TKT-069",
        "category": "Password",
        "issue": "Shared training-room account password expired the morning of a new "
                 "hire orientation session, locking out the whole onboarding class.",
        "resolution": "Reset the shared account password immediately, set it to never "
                       "expire per the room's existing exception policy which had lapsed, "
                       "and documented the account in the password vault for future renewal "
                       "reminders so the cohort could continue their session.",
        "resolved_in_minutes": 15,
    },
    {
        "ticket_id": "TKT-070",
        "category": "Password",
        "issue": "User changes their Okta password but their on-prem AD password stays "
                 "the old one, causing confusion about which password works where.",
        "resolution": "Identified that the Okta-to-AD password write-back agent had been "
                       "disabled during a recent server patch. Re-enabled the write-back "
                       "agent, had the user perform one more password change to sync both "
                       "directions, and confirmed Okta and AD passwords matched.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-071",
        "category": "Password",
        "issue": "Overnight MFA app update wiped enrolled accounts for several users, who "
                 "now cannot approve push notifications to reset their passwords.",
        "resolution": "Confirmed the vendor update had reset local token storage on "
                       "affected phones. Issued temporary bypass codes for each impacted "
                       "user, had them re-enroll Microsoft Authenticator from scratch, and "
                       "verified self-service password reset worked again.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-072",
        "category": "Password",
        "issue": "Database admin's privileged account password needs rotation per policy "
                 "but the credential vault shows it as 'checked out' and won't let them rotate it.",
        "resolution": "Filed the required change-control ticket, then worked with the "
                       "vault administrator to force-release the stale checkout left from a "
                       "crashed session. Rotated the password through the vault, updated "
                       "the linked service accounts, and confirmed connectivity.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-073",
        "category": "Password",
        "issue": "Security monitoring flagged a user's account for impossible-travel "
                 "logins minutes apart from two countries; user denies traveling.",
        "resolution": "Escalated immediately to the Tier 2 security team on suspicion of "
                       "compromise. They force-reset the password, revoked all active "
                       "sessions and refresh tokens, re-enabled MFA enrollment from "
                       "scratch, and confirmed no further suspicious sign-ins occurred.",
        "resolved_in_minutes": 90,
    },
    {
        "ticket_id": "TKT-074",
        "category": "Password",
        "issue": "Temporary contractor cannot remember their badge PIN at the building "
                 "kiosk on their first day and has no manager present to verify identity.",
        "resolution": "Verified the contractor's identity against their signed engagement "
                       "letter and photo ID at the front desk, reset the kiosk PIN in the "
                       "visitor management system, and confirmed they could badge in and "
                       "log in to their loaner laptop.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-075",
        "category": "Password",
        "issue": "User cannot use self-service password reset because their security "
                 "questions were set up years ago and they no longer remember the answers.",
        "resolution": "Since SSPR verification failed, performed manual identity "
                       "verification via manager confirmation and badge photo lookup, reset "
                       "the password directly in AD, and helped the user update their "
                       "security questions to ones they would actually remember.",
        "resolved_in_minutes": 25,
    },

    # ------------------------------------------------- Software Access (cont.)
    {
        "ticket_id": "TKT-076",
        "category": "Software Access",
        "issue": "New hire's onboarding ticket bundles five software requests at once: "
                 "Microsoft 365, Teams, SAP, Adobe Acrobat, and a department reporting tool.",
        "resolution": "Worked through each request in sequence: assigned the M365 E3 "
                       "license, added the user to Teams and the SAP role group, deployed "
                       "Adobe Acrobat via SCCM, and granted reporting-tool access after "
                       "manager approval. Confirmed all five applications opened correctly.",
        "resolved_in_minutes": 90,
    },
    {
        "ticket_id": "TKT-077",
        "category": "Software Access",
        "issue": "After an Adobe Acrobat auto-update, several users can no longer sign in "
                 "through the company SSO plugin and are prompted for a local password instead.",
        "resolution": "The update had replaced the SSO-enabled installer with the consumer "
                       "version. Uninstalled the consumer build, redeployed the "
                       "enterprise MSI with the SSO plugin preconfigured via SCCM, and "
                       "disabled in-app auto-update to prevent recurrence.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-078",
        "category": "Software Access",
        "issue": "User's Power BI workspace access disappeared after a department reorg, "
                 "even though their on-prem AD group membership looks correct.",
        "resolution": "Found the Azure AD Connect sync job had failed to push the updated "
                       "group membership for several days. Forced a manual delta sync, "
                       "verified the security group reflected in Azure AD, and confirmed "
                       "the user regained access to the Power BI workspace.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-079",
        "category": "Software Access",
        "issue": "Entire design team loses AutoCAD access mid-day with 'license server "
                 "unreachable,' and Tier 1 cannot restart the licensing service remotely.",
        "resolution": "Escalated to Tier 2 infrastructure, who found the license server VM "
                       "had run out of disk space and crashed. Freed disk space, restarted "
                       "the Autodesk licensing service, and confirmed all design seats "
                       "checked out licenses again.",
        "resolved_in_minutes": 65,
    },
    {
        "ticket_id": "TKT-080",
        "category": "Software Access",
        "issue": "Engineer needs the Visio add-on beyond their standard Microsoft 365 "
                 "plan, which requires separate budget approval before it can be assigned.",
        "resolution": "Submitted the Visio Plan 2 request with finance approval attached "
                       "in ServiceNow, waited for budget sign-off, then assigned the add-on "
                       "license and pushed the desktop app via SCCM once approved, "
                       "confirming the engineer could open and edit diagrams.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-081",
        "category": "Software Access",
        "issue": "Short-term contractor needs scoped application access without a full "
                 "employee AD account, since standard onboarding doesn't fit their engagement.",
        "resolution": "Provisioned a guest-tier Azure AD B2B account scoped to only the "
                       "two applications required for the engagement, set an automatic "
                       "expiration matching the contract end date, and confirmed the "
                       "contractor could sign in without standard employee-level access.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-082",
        "category": "Software Access",
        "issue": "After upgrading to Windows 11, a legacy in-house quality app fails to "
                 "launch with a 'missing runtime component' error for several shop-floor users.",
        "resolution": "Identified that the old .NET runtime the app depended on was "
                       "removed during the Windows 11 upgrade. Packaged and deployed the "
                       "required legacy runtime alongside compatibility-mode settings via "
                       "SCCM, and confirmed the quality app launched normally again.",
        "resolved_in_minutes": 70,
    },
    {
        "ticket_id": "TKT-083",
        "category": "Software Access",
        "issue": "After a department reorg, a user's SharePoint permissions and Teams "
                 "membership are now inconsistent, giving access to one but not the other.",
        "resolution": "Found the reorg script had updated the Teams owner list but missed "
                       "the underlying SharePoint site permissions inherited separately. "
                       "Manually aligned the SharePoint group membership to match the new "
                       "team roster and confirmed both systems showed consistent access.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-084",
        "category": "Software Access",
        "issue": "Minitab license server crashes repeatedly every few hours, kicking out "
                 "all connected quality engineers and requiring daily manual restarts.",
        "resolution": "Escalated to Tier 2 and the vendor's support line after repeated "
                       "crashes. Together they identified a memory leak in the current "
                       "license manager build, applied a vendor hotfix, and confirmed "
                       "stability held over a full week of monitoring.",
        "resolved_in_minutes": 100,
    },
    {
        "ticket_id": "TKT-085",
        "category": "Software Access",
        "issue": "Analytics team wants a new BI tool rolled out company-wide, but it first "
                 "needs a security review before any production access can be granted.",
        "resolution": "Coordinated a phased rollout: submitted the tool for security and "
                       "data-privacy review, ran a two-week sandboxed pilot with a small "
                       "group once approved, then expanded license assignment to the full "
                       "analytics team after the pilot completed without issues.",
        "resolved_in_minutes": 120,
    },
    {
        "ticket_id": "TKT-086",
        "category": "Software Access",
        "issue": "After a Chrome update, users can no longer use the legacy web-based ERP "
                 "front end, which depends on an old Flash-era plugin Chrome now blocks.",
        "resolution": "Confirmed Chrome had removed support for the legacy plugin entirely. "
                       "Deployed Microsoft Edge with IE mode enabled via group policy as a "
                       "supported workaround for the legacy ERP front end, and set it as the "
                       "default browser shortcut on affected workstations.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-087",
        "category": "Software Access",
        "issue": "A temporary elevated-admin request for a new hire was mistakenly routed "
                 "to the wrong manager for approval and has sat untouched for three days.",
        "resolution": "Identified the routing error in the ServiceNow approval workflow, "
                       "manually reassigned the request to the correct manager, followed up "
                       "to get same-day sign-off, and provisioned the time-limited elevated "
                       "access once approved.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-088",
        "category": "Software Access",
        "issue": "A license-reclamation script meant to remove inactive accounts "
                 "accidentally stripped a currently active employee's Office license.",
        "resolution": "Found the script's inactivity threshold had misread a recent leave "
                       "of absence as permanent inactivity. Reassigned the Microsoft 365 "
                       "license immediately, confirmed the user's data and mailbox were "
                       "intact, and excluded leave-of-absence accounts from the script.",
        "resolved_in_minutes": 25,
    },

    # ------------------------------------------------------------ Hardware (cont.)
    {
        "ticket_id": "TKT-089",
        "category": "Hardware",
        "issue": "New hire's laptop was supposed to be pre-imaged and ready for their "
                 "start date but is still sitting in the imaging queue the morning they arrive.",
        "resolution": "Expedited the imaging job ahead of the queue, applied the standard "
                       "build along with the required department software bundle, and had "
                       "the laptop ready and domain-joined within the first hour so the new "
                       "hire could begin onboarding only slightly delayed.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-090",
        "category": "Hardware",
        "issue": "User's external monitor, dock, and laptop all seem to have issues at "
                 "once: flickering display, dock not charging, and laptop randomly restarting.",
        "resolution": "Isolated the issues one at a time: a failing dock power adapter "
                       "explained both the charging problem and restarts under load, while "
                       "a loose HDMI cable caused the flicker. Replaced the adapter, reseated "
                       "the cable, and confirmed all three symptoms were resolved.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-091",
        "category": "Hardware",
        "issue": "After a vendor BIOS update, the fingerprint reader on several laptops "
                 "stopped working for Windows Hello sign-in.",
        "resolution": "Confirmed the BIOS update had reset the TPM and broken the existing "
                       "Windows Hello enrollment. Cleared the TPM, reinstalled the "
                       "fingerprint reader driver matched to the new BIOS version, and had "
                       "each affected user re-enroll their fingerprint successfully.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-092",
        "category": "Hardware",
        "issue": "A user's laptop has blue-screened five times this week with different "
                 "error codes each time, despite a clean OS reinstall by Tier 1.",
        "resolution": "Escalated to Tier 2 hardware diagnostics, who ran extended memory "
                       "diagnostics overnight and identified a failing RAM module. Replaced "
                       "the module under the manufacturer's warranty and confirmed no "
                       "further crashes after a week of monitoring.",
        "resolved_in_minutes": 90,
    },
    {
        "ticket_id": "TKT-093",
        "category": "Hardware",
        "issue": "User's Teams calls freeze completely whenever their laptop is docked, "
                 "but work fine when undocked and running on battery.",
        "resolution": "Traced the freezing to the docking station's USB controller "
                       "conflicting with the laptop's power-management settings under load. "
                       "Updated the dock firmware and disabled USB selective suspend in "
                       "power settings, which resolved the freezing while docked.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-094",
        "category": "Hardware",
        "issue": "User's laptop needs to go in for a multi-day repair, but they have a "
                 "critical client presentation tomorrow and no backup device.",
        "resolution": "Issued a loaner laptop from the IT spares pool same-day, restored "
                       "the user's profile and key files from their OneDrive backup, and "
                       "confirmed they had everything needed for the presentation while "
                       "their primary laptop went out for repair.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-095",
        "category": "Hardware",
        "issue": "Following a Windows update rollout, Bluetooth mice and headsets stopped "
                 "working across a large portion of the office fleet simultaneously.",
        "resolution": "Confirmed the update had shipped a broken generic Bluetooth driver. "
                       "Rolled back the driver to the previous known-good version fleet-wide "
                       "via SCCM and blocked the problematic driver update from "
                       "redeploying until the vendor issued a fix.",
        "resolved_in_minutes": 80,
    },
    {
        "ticket_id": "TKT-096",
        "category": "Hardware",
        "issue": "A workstation's wired network connection drops every few minutes, and "
                 "swapping cables and switch ports has not made any difference.",
        "resolution": "Ran extended diagnostics on the workstation itself and found the "
                       "onboard NIC was intermittently failing. Disabled the onboard NIC, "
                       "installed a USB-to-Ethernet adapter as a stopgap, and ordered a "
                       "motherboard replacement for a scheduled future repair.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-097",
        "category": "Hardware",
        "issue": "A ruggedized tablet used on the shop floor has failed for the third time "
                 "this quarter with the same cracked-screen pattern despite protective cases.",
        "resolution": "Escalated the recurring failure to the OEM under warranty for an "
                       "RMA and root-cause review. The vendor identified a batch defect in "
                       "the screen adhesive, replaced the unit with a corrected model, and "
                       "issued replacement cases rated for the heavier shop-floor impact.",
        "resolved_in_minutes": 75,
    },
    {
        "ticket_id": "TKT-098",
        "category": "Hardware",
        "issue": "New ultra-wide monitors purchased for the team are not displaying at "
                 "full resolution through the existing docking stations.",
        "resolution": "Determined the older dock model's DisplayPort output did not "
                       "support the monitor's native resolution and refresh rate. Updated "
                       "the dock firmware where possible and ordered current-generation "
                       "docks for the units that still could not drive full resolution.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-099",
        "category": "Hardware",
        "issue": "New hire with an approved ergonomic accommodation needs a standing desk "
                 "converter and an external monitor set up before their start date.",
        "resolution": "Coordinated with facilities to install the standing desk converter "
                       "and worked with the new hire's manager to confirm monitor and dock "
                       "specifications from the accommodation request, then set up and "
                       "tested the full workstation a day before their start date.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-100",
        "category": "Hardware",
        "issue": "After a macOS update, a designer's MacBook can no longer connect to the "
                 "corporate AnyConnect VPN client, citing a kernel extension permission error.",
        "resolution": "The macOS update had revoked the legacy kernel extension's system "
                       "approval. Reinstalled the updated AnyConnect client built on the "
                       "newer system-extension framework and walked the user through "
                       "re-approving it in System Settings, restoring VPN connectivity.",
        "resolved_in_minutes": 45,
    },

    # ---------------------------------------------------------------- Email (cont.)
    {
        "ticket_id": "TKT-101",
        "category": "Email",
        "issue": "New hire's mailbox was created but they are missing from three "
                 "department distribution lists they need for day-one communications.",
        "resolution": "Cross-checked the onboarding checklist against the distribution "
                       "list owners, added the new hire to all three lists in the Exchange "
                       "admin center, and confirmed they began receiving department "
                       "announcements within the hour.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-102",
        "category": "Email",
        "issue": "After the latest Outlook version rollout, several users' shared team "
                 "calendars no longer overlay correctly, showing only one calendar at a time.",
        "resolution": "Confirmed the new Outlook version had changed how overlay mode is "
                       "toggled per calendar group. Reconfigured the affected users' "
                       "calendar groups and overlay settings, and published a quick "
                       "how-to guide since the toggle had moved in the new layout.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-103",
        "category": "Email",
        "issue": "Emailed documents that used to auto-save to a SharePoint library now "
                 "fail silently after a recent platform migration.",
        "resolution": "Found the integration's stored connection still pointed at the old "
                       "SharePoint tenant URL after migration. Updated the connector "
                       "configuration to the new tenant endpoint, re-authenticated the "
                       "service account, and confirmed documents saved correctly again.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-104",
        "category": "Email",
        "issue": "A manager's mailbox shows signs of compromise: sent items contain "
                 "phishing emails they never wrote, sent to external addresses overnight.",
        "resolution": "Escalated immediately to the Tier 2 security incident response "
                       "team. They disabled the account, revoked all active sessions and "
                       "app passwords, force-reset credentials, and reviewed mailbox rules "
                       "for a malicious auto-forward, which they found and removed.",
        "resolved_in_minutes": 100,
    },
    {
        "ticket_id": "TKT-105",
        "category": "Email",
        "issue": "During the phased migration from on-prem Exchange to Exchange Online, "
                 "a batch of users report missing emails from the days right before cutover.",
        "resolution": "Traced the gap to a migration batch that completed before the final "
                       "incremental sync captured the last day of on-prem mail. Reran the "
                       "incremental sync for the affected mailboxes and confirmed the "
                       "missing messages appeared after the batch completed.",
        "resolved_in_minutes": 85,
    },
    {
        "ticket_id": "TKT-106",
        "category": "Email",
        "issue": "New hire's onboarding ticket requires access to two shared department "
                 "mailboxes plus their personal signature template before their first day.",
        "resolution": "Granted Full Access and Send As on both shared mailboxes in the "
                       "Exchange admin center, deployed the standard department signature "
                       "template via the email signature tool, and confirmed everything "
                       "appeared correctly when the new hire logged in.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-107",
        "category": "Email",
        "issue": "Since the latest macOS Mail update, several Mac users can no longer "
                 "sync their corporate Exchange mailboxes; the app reports a certificate error.",
        "resolution": "Confirmed the OS update had changed its certificate trust handling "
                       "for ActiveSync. Reissued the device certificates through the MDM "
                       "profile, had affected users remove and re-add their Exchange "
                       "accounts, and verified mail synced normally afterward.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-108",
        "category": "Email",
        "issue": "Meetings scheduled in Microsoft Teams are not appearing on some users' "
                 "Outlook calendars, even though the Teams invite shows as sent.",
        "resolution": "Identified a stuck Exchange Web Services connection on the affected "
                       "mailboxes preventing the Teams add-in from writing calendar items. "
                       "Repaired the Outlook profile and re-authenticated the Teams add-in, "
                       "which restored calendar sync for new meeting invites.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-109",
        "category": "Email",
        "issue": "Outbound email to one specific partner company has been delayed by "
                 "hours for two days, while all other mail flow looks normal.",
        "resolution": "Escalated to the Tier 2 messaging team, who found a misconfigured "
                       "connector throttling policy that was specifically rate-limiting mail "
                       "to that partner's domain after a prior spam incident. Corrected the "
                       "throttle setting and confirmed delivery times returned to normal.",
        "resolved_in_minutes": 70,
    },
    {
        "ticket_id": "TKT-110",
        "category": "Email",
        "issue": "Legal has requested a litigation hold and full mailbox export for an "
                 "employee involved in an active investigation.",
        "resolution": "Verified the request through the legal department's formal process, "
                       "placed the mailbox on litigation hold in the Microsoft Purview "
                       "compliance center, and exported the requested mailbox content to a "
                       "secure location for legal's review, logging the chain of custody.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-111",
        "category": "Email",
        "issue": "New hire needs their out-of-office reply and email signature configured "
                 "to match company branding standards before their first official client email.",
        "resolution": "Applied the standard branded signature template through the central "
                       "signature management tool and configured a default out-of-office "
                       "template the new hire could customize, confirming both rendered "
                       "correctly on a test message.",
        "resolved_in_minutes": 15,
    },
    {
        "ticket_id": "TKT-112",
        "category": "Email",
        "issue": "After updating the Outlook mobile app, a user can no longer sync mail "
                 "on their phone; the app reports a certificate-based authentication failure.",
        "resolution": "The update had changed how the app handles client certificates "
                       "issued by the corporate MDM. Reissued the device's mail profile "
                       "certificate through Intune, removed and re-added the account in "
                       "the mobile app, and confirmed mail synced again.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-113",
        "category": "Email",
        "issue": "After two companies' tenants were merged, free/busy availability does "
                 "not show correctly when scheduling meetings between the two former organizations.",
        "resolution": "Found the cross-tenant organization relationship for free/busy "
                       "sharing had not been configured as part of the merger cutover. Set "
                       "up the organization relationship and sharing policy between both "
                       "tenants, and confirmed availability displayed correctly across both.",
        "resolved_in_minutes": 65,
    },

    # -------------------------------------------------------------- Printer (cont.)
    {
        "ticket_id": "TKT-114",
        "category": "Printer",
        "issue": "New hire's badge was issued today but cannot release any held print "
                 "jobs at the secure printer; the system says 'badge not recognized.'",
        "resolution": "The badge office's nightly export to the print management system "
                       "had not yet included the new hire's badge ID. Manually added the "
                       "badge number to the print system console and confirmed a held job "
                       "released successfully on the next tap.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-115",
        "category": "Printer",
        "issue": "A failed multifunction printer is being replaced, but the new unit has "
                 "a different driver and the old print queue, scan profiles, and shortcuts all need migrating.",
        "resolution": "Removed the old printer object and driver from the print server, "
                       "installed the new model's driver, recreated the shared queue under "
                       "the same name so existing shortcuts kept working, and rebuilt the "
                       "scan-to-folder profiles on the new device.",
        "resolved_in_minutes": 75,
    },
    {
        "ticket_id": "TKT-116",
        "category": "Printer",
        "issue": "After a print driver update was pushed fleet-wide, duplex printing "
                 "silently stopped working and everything now prints single-sided.",
        "resolution": "Confirmed the new driver version had reset the default duplex "
                       "setting on every queue it touched. Rolled back to the previous "
                       "validated driver version on the print server and reapplied the "
                       "duplex-by-default policy, restoring two-sided printing fleet-wide.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-117",
        "category": "Printer",
        "issue": "The main print server has crashed three times this week, taking down "
                 "every networked printer in the building each time.",
        "resolution": "Escalated to the Tier 2 server team, who found a memory leak in the "
                       "print spooler service triggered by a corrupted driver package. "
                       "Removed the offending driver, applied a Microsoft hotfix for the "
                       "spooler issue, and monitored stability over several days.",
        "resolved_in_minutes": 95,
    },
    {
        "ticket_id": "TKT-118",
        "category": "Printer",
        "issue": "Scan-to-email from the office multifunction printer stopped working "
                 "after the email system's connector settings changed; scans never arrive.",
        "resolution": "Found the printer's SMTP relay settings still referenced the "
                       "decommissioned on-prem relay after the migration to a new "
                       "connector. Updated the printer's SMTP configuration to the new "
                       "authenticated relay endpoint and confirmed test scans delivered.",
        "resolved_in_minutes": 40,
    },
    {
        "ticket_id": "TKT-119",
        "category": "Printer",
        "issue": "Three departments are getting new color printers installed this week, "
                 "each needing drivers packaged and queues set up before go-live.",
        "resolution": "Packaged the vendor's universal driver for SCCM deployment, created "
                       "and shared the queues on the print server for each department, "
                       "scheduled the physical installs with facilities, and confirmed test "
                       "pages printed successfully at each location before cutover.",
        "resolved_in_minutes": 90,
    },
    {
        "ticket_id": "TKT-120",
        "category": "Printer",
        "issue": "Remote employee wants to print directly from their work laptop to their "
                 "home printer through the VPN print server but cannot find it in the list.",
        "resolution": "Explained that home printers are not supported through the "
                       "corporate VPN print server for security reasons, and instead set "
                       "the user up with a virtual PDF printer plus instructions to print "
                       "locally over their own home network once the PDF was generated.",
        "resolved_in_minutes": 20,
    },
    {
        "ticket_id": "TKT-121",
        "category": "Printer",
        "issue": "A network firmware update on a multifunction printer caused it to lose "
                 "its static IP and drop off the network entirely afterward.",
        "resolution": "Confirmed the firmware update had reset network settings to DHCP "
                       "defaults. Reconfigured the printer with its assigned static IP and "
                       "DNS settings, updated the print server's port to match, and "
                       "confirmed the device came back online for all users.",
        "resolved_in_minutes": 35,
    },
    {
        "ticket_id": "TKT-122",
        "category": "Printer",
        "issue": "Engineering's large-format plotter has a grinding noise from the roll "
                 "motor and has stopped advancing media partway through every job.",
        "resolution": "Escalated to the vendor's field technician after Tier 1 confirmed "
                       "it was a hardware failure beyond standard troubleshooting. The "
                       "technician replaced the worn roll-advance motor under the service "
                       "contract and verified several full-length test plots printed cleanly.",
        "resolved_in_minutes": 120,
    },
    {
        "ticket_id": "TKT-123",
        "category": "Printer",
        "issue": "After the HR department rolled out new employee badges, the secure "
                 "print release system stopped recognizing any badge company-wide.",
        "resolution": "Found the new badge format used a different encoding that the "
                       "print release software's older firmware did not support. Updated "
                       "the print release server's badge-reader firmware and reimported the "
                       "badge database, restoring badge-based release for all employees.",
        "resolved_in_minutes": 80,
    },
    {
        "ticket_id": "TKT-124",
        "category": "Printer",
        "issue": "New print quota policy rolled out company-wide is blocking several "
                 "legitimate high-volume users, like the print shop, from printing at all.",
        "resolution": "Reviewed the quota policy's default thresholds against business "
                       "needs and identified that print-shop accounts had been included by "
                       "mistake. Added a quota exception group for high-volume business "
                       "accounts and confirmed normal printing resumed for those teams.",
        "resolved_in_minutes": 45,
    },
    {
        "ticket_id": "TKT-125",
        "category": "Printer",
        "issue": "Temporary contractor with a visitor badge cannot release any print jobs "
                 "on their first day since visitor badges aren't normally enrolled in the print system.",
        "resolution": "Coordinated with the badge office to issue a contractor-tier badge "
                       "instead of a standard visitor badge, enrolled it in the secure print "
                       "system with department-level permissions, and confirmed the "
                       "contractor could release jobs at the shared printer.",
        "resolved_in_minutes": 30,
    },

    # ------------------------------------------------------------- Network (cont.)
    {
        "ticket_id": "TKT-126",
        "category": "Network",
        "issue": "New hire's desk has no network connectivity at all on their first day; "
                 "the port shows as 'unauthorized' when their laptop is plugged in.",
        "resolution": "Confirmed the desk's switch port had not been added to the correct "
                       "VLAN as part of the new-hire workspace setup. Updated the port "
                       "configuration to the standard data VLAN, reauthorized it in NAC, "
                       "and verified the new hire's laptop connected with full access.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-127",
        "category": "Network",
        "issue": "After a scheduled core switch firmware update overnight, an entire "
                 "floor lost access to several internal applications while internet still worked.",
        "resolution": "Found the firmware update had reset several VLAN tag assignments "
                       "on trunk ports to defaults. Reapplied the documented VLAN "
                       "configuration from the pre-change backup, verified trunk tagging "
                       "matched the network diagram, and confirmed all applications were reachable.",
        "resolved_in_minutes": 85,
    },
    {
        "ticket_id": "TKT-128",
        "category": "Network",
        "issue": "Following a firewall rule change for a security audit, Finance users "
                 "can no longer reach both SAP and the department file share simultaneously.",
        "resolution": "Reviewed the new rule set and found it had inadvertently scoped the "
                       "Finance subnet too narrowly, excluding both the SAP application "
                       "servers and file share IP ranges. Corrected the rule to include both "
                       "required ranges and confirmed access was restored to both systems.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-129",
        "category": "Network",
        "issue": "Users at the secondary site report consistently high latency to the main "
                 "data center, occasionally over a full second, slowing every application.",
        "resolution": "Escalated to Tier 2 network engineering, who ran continuous path "
                       "traces and found a saturated link on the site-to-site WAN circuit "
                       "during business hours. Worked with the carrier to upgrade the "
                       "circuit's bandwidth and confirmed latency dropped to normal levels.",
        "resolved_in_minutes": 130,
    },
    {
        "ticket_id": "TKT-130",
        "category": "Network",
        "issue": "New conference room build-out needs network drops, Wi-Fi coverage "
                 "validation, and AV equipment all connected and tested before the room opens.",
        "resolution": "Coordinated with facilities to terminate the new network drops, "
                       "configured switch ports for the room's AV controller and camera, "
                       "ran a Wi-Fi site survey to confirm adequate coverage, and tested a "
                       "full Teams meeting from the room before handing it over.",
        "resolved_in_minutes": 100,
    },
    {
        "ticket_id": "TKT-131",
        "category": "Network",
        "issue": "Contractor team starting tomorrow needs temporary wired network access "
                 "in a shared workspace that is normally only provisioned for badge-only guest Wi-Fi.",
        "resolution": "Set up a temporary isolated VLAN for the contractor workspace with "
                       "internet-only access and no internal routing, configured the "
                       "relevant switch ports, and confirmed each contractor's laptop could "
                       "connect with appropriately restricted access on day one.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-132",
        "category": "Network",
        "issue": "Since a wireless controller firmware update, access points across the "
                 "building intermittently reboot, dropping all connected users for a few minutes at a time.",
        "resolution": "Confirmed the firmware update had a known stability bug affecting "
                       "AP-to-controller heartbeats. Rolled back the controller firmware to "
                       "the prior stable version, applied the vendor's recommended interim "
                       "patch, and monitored AP uptime over several days without further drops.",
        "resolved_in_minutes": 75,
    },
    {
        "ticket_id": "TKT-133",
        "category": "Network",
        "issue": "After a domain controller failure, users across the site cannot get new "
                 "DHCP leases or resolve internal hostnames, though existing connections still work.",
        "resolution": "Escalated to Tier 2 infrastructure, who confirmed the failed domain "
                       "controller was also hosting the site's DHCP and DNS roles with no "
                       "active failover configured. Promoted the secondary domain "
                       "controller to take over both roles and restored full connectivity.",
        "resolved_in_minutes": 90,
    },
    {
        "ticket_id": "TKT-134",
        "category": "Network",
        "issue": "Intermittent connectivity issues to a cloud provider have been "
                 "occurring for two days, with traceroutes showing erratic paths each time.",
        "resolution": "Escalated to the Tier 2 network team and the upstream ISP, who "
                       "identified a flapping BGP route at the WAN edge causing the erratic "
                       "paths. The ISP corrected the route advertisement and engineering "
                       "confirmed stable, consistent routing afterward.",
        "resolved_in_minutes": 105,
    },
    {
        "ticket_id": "TKT-135",
        "category": "Network",
        "issue": "An office move over the weekend requires re-cabling, re-IP'ing, and "
                 "re-labeling every desk and printer port before staff return Monday.",
        "resolution": "Worked with facilities and a cabling contractor through the weekend "
                       "to run new cabling, reassign IP addresses and VLANs to match the new "
                       "floor plan, relabel every port in the patch panel, and tested every "
                       "desk and printer connection before staff arrived Monday morning.",
        "resolved_in_minutes": 110,
    },
    {
        "ticket_id": "TKT-136",
        "category": "Network",
        "issue": "New hire's laptop connects to the network physically but gets no IP "
                 "address and cannot reach anything, even though the cable tests fine.",
        "resolution": "Found the desk's switch port had been left in the default "
                       "quarantine VLAN used for unprovisioned ports. Reassigned the port "
                       "to the correct departmental VLAN per the seating chart, and "
                       "confirmed the laptop picked up a valid IP address and connectivity.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-137",
        "category": "Network",
        "issue": "After a VPN concentrator firmware update, the site-to-site tunnel to a "
                 "remote warehouse keeps dropping and re-establishing every few minutes.",
        "resolution": "Confirmed the firmware update had changed default IKE rekey "
                       "intervals, causing a mismatch with the remote site's older "
                       "concentrator. Aligned the rekey and lifetime settings on both ends "
                       "to match, and confirmed the tunnel stayed stable overnight.",
        "resolved_in_minutes": 70,
    },
    {
        "ticket_id": "TKT-138",
        "category": "Network",
        "issue": "A core switch failure has taken down both file sharing and print "
                 "services for an entire department simultaneously.",
        "resolution": "Escalated to Tier 2, who failed traffic over to the redundant core "
                       "switch while diagnosing the primary unit, which had suffered a "
                       "power supply failure. Replaced the power supply, restored the "
                       "primary switch to service, and confirmed both services were back online.",
        "resolved_in_minutes": 80,
    },

    # --------------------------------------------------------- ERP Access (cont.)
    {
        "ticket_id": "TKT-139",
        "category": "ERP Access",
        "issue": "New buyer's onboarding requires a full purchasing role bundle in SAP "
                 "covering requisitions, purchase orders, and vendor master display.",
        "resolution": "Submitted the bundled role request through ServiceNow, confirmed no "
                       "segregation-of-duties conflicts in GRC, and assigned the combined "
                       "purchasing composite role in SU01. Verified the new buyer could "
                       "create requisitions and view vendor master records on day one.",
        "resolved_in_minutes": 60,
    },
    {
        "ticket_id": "TKT-140",
        "category": "ERP Access",
        "issue": "A planner's role change request needs GRC review, manager sign-off, and "
                 "final SU01 assignment before they can access the new planning transactions.",
        "resolution": "Walked the request through each required stage: GRC flagged no "
                       "conflicts, the manager approved in ServiceNow, and the new role was "
                       "assigned in SU01. Confirmed with the planner that the new planning "
                       "transactions opened correctly after the user buffer refreshed.",
        "resolved_in_minutes": 50,
    },
    {
        "ticket_id": "TKT-141",
        "category": "ERP Access",
        "issue": "After the latest SAP support package was applied, a custom Z-report used "
                 "by the finance team throws a syntax error and no longer runs.",
        "resolution": "Escalated to the Tier 2 SAP Basis team, who found the support "
                       "package had deprecated a function module the custom report relied "
                       "on. The ABAP team patched the report to use the supported "
                       "replacement module and confirmed it ran cleanly in production.",
        "resolved_in_minutes": 95,
    },
    {
        "ticket_id": "TKT-142",
        "category": "ERP Access",
        "issue": "A user's SAP session repeatedly produces a short dump with 'TSV_TNEW_PAGE_ALLOC_FAILED' "
                 "whenever they run a large inventory report.",
        "resolution": "Escalated to Tier 2 SAP Basis, who identified the application server "
                       "was running low on extended memory during peak hours. Increased the "
                       "memory parameters for the relevant work processes and had the user "
                       "rerun the report successfully without a dump.",
        "resolved_in_minutes": 80,
    },
    {
        "ticket_id": "TKT-143",
        "category": "ERP Access",
        "issue": "The EDI integration between SAP and a key supplier stopped processing "
                 "purchase order acknowledgments after a recent SAP update.",
        "resolution": "Found the update had altered an IDoc segment structure the EDI "
                       "middleware was not expecting, causing silent processing failures. "
                       "Updated the middleware mapping to match the new IDoc structure and "
                       "confirmed acknowledgments began processing normally again.",
        "resolved_in_minutes": 90,
    },
    {
        "ticket_id": "TKT-144",
        "category": "ERP Access",
        "issue": "New hire's permanent SAP role is still pending manager approval, but "
                 "they need basic display access today to begin training.",
        "resolution": "Granted a temporary display-only role with no posting authority "
                       "while the full role request remained in approval, set it to expire "
                       "automatically in two weeks, and confirmed the new hire could follow "
                       "along in training sessions using read-only access.",
        "resolved_in_minutes": 25,
    },
    {
        "ticket_id": "TKT-145",
        "category": "ERP Access",
        "issue": "After upgrading SAP GUI to the latest version, a finance user's Excel "
                 "macro that pulls data via SAP's RFC interface stopped connecting entirely.",
        "resolution": "Found the new SAP GUI version had changed default RFC library paths "
                       "the macro depended on. Reinstalled the SAP GUI with the legacy RFC "
                       "SDK components included and updated the macro's connection string, "
                       "restoring the automated data pull.",
        "resolved_in_minutes": 65,
    },
    {
        "ticket_id": "TKT-146",
        "category": "ERP Access",
        "issue": "A cross-functional employee needs both MM and SD module access in SAP "
                 "for a new role spanning purchasing and sales order management.",
        "resolution": "Submitted the combined MM and SD role request for segregation-of-"
                       "duties review in GRC, which flagged a minor conflict requiring a "
                       "mitigating control. Documented the control, obtained sign-off, and "
                       "assigned both roles once approved.",
        "resolved_in_minutes": 70,
    },
    {
        "ticket_id": "TKT-147",
        "category": "ERP Access",
        "issue": "SAP production system performance has degraded noticeably for all users "
                 "over the past two days, with simple transactions taking several seconds longer.",
        "resolution": "Escalated to the Tier 2 Basis and DBA team, who found a long-running "
                       "batch job had been accidentally scheduled during business hours and "
                       "was consuming excessive database resources. Rescheduled the job to "
                       "off-hours and confirmed performance returned to normal.",
        "resolved_in_minutes": 75,
    },
    {
        "ticket_id": "TKT-148",
        "category": "ERP Access",
        "issue": "The integration between SAP and Concur for expense report submission "
                 "has been failing silently, leaving employees' expense reports stuck in draft.",
        "resolution": "Found an expired service account password used by the SAP-Concur "
                       "integration after a routine password rotation. Updated the "
                       "credentials in the integration configuration, reprocessed the "
                       "stuck batch of expense reports, and confirmed new submissions flowed through.",
        "resolved_in_minutes": 55,
    },
    {
        "ticket_id": "TKT-149",
        "category": "ERP Access",
        "issue": "An employee who completed onboarding still has the broad temporary SAP "
                 "access granted on day one and now needs it scoped down to their permanent role.",
        "resolution": "Reviewed the employee's confirmed permanent role assignment, removed "
                       "the temporary broad-access role in SU01, and assigned only the "
                       "authorizations matching their actual job function, confirming with "
                       "the employee that all needed transactions still worked.",
        "resolved_in_minutes": 30,
    },
    {
        "ticket_id": "TKT-150",
        "category": "ERP Access",
        "issue": "During a production issue at month-end close, a controller needs "
                 "emergency firefighter access to a restricted finance transaction outside their normal role.",
        "resolution": "Processed the emergency access request through the firefighter ID "
                       "workflow, requiring incident justification and post-use audit log "
                       "review. Granted temporary access for the duration of the incident, "
                       "and confirmed the audit log was reviewed and access revoked afterward.",
        "resolved_in_minutes": 40,
    },
]
