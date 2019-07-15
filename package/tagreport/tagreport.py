class tagreport:

    def __init__(self, PR_rev):

        """ defines tagreport based on PR and rev number

        parameters: PR_rev (str)
        attributes: PR (str)
                 rev (int)

        """
        
        self.PR = PR_rev.split('_')[0]
        self.rev = int(PR_rev.split('_')[1])


    def tag_query(self):
    
        selectstr1 = "(select distinct F.name as CustomerName, F.Customerid, B.name as LDC_Account, B.Accountid,"
        selectstr1 = "".join([selectstr1, "D.uidaccount, D.marketcode, A.Contractid, A.Revision "])
        selectstr1 = "".join([selectstr1, "from pwrline.account B, pwrline.lscmcontract A, pwrline.lscmcontractitem C, "])
        selectstr1 = "".join([selectstr1, "pwrline.acctservicehist D, pwrline.customer F "])
        selectstr1 = "".join([selectstr1, "where C.uidcontract=A.uidcontract and C.uidaccount=B.uidaccount and B.uidaccount=D.uidaccount "])
        selectstr1 = "".join([selectstr1, "and B.uidcustomer=F.uidcustomer and A.contractid='", self.PR, "' and A.revision=", rev_num, ") A "])

        s2str = "select distinct A.*, B.starttime, B.stoptime, B.overridecode as Tag_Type, B.val as Tag, B.strval as SOURCE_TYPE, B.lstime as Timestamp "
        s2str = "".join([s2str, "from pwrline.acctoverridehist B, ", selectstr1])
        s2str = "".join([s2str, "where A.uidaccount=B.uidaccount and (A.marketcode='PJM' OR  A.marketcode='NEPOOL' OR A.marketcode= 'NYISO' OR A.marketcode= 'MISO') "])
        s2str = "".join([s2str, "and (B.overridecode ='TRANSMISSION_TAG_OVRD' OR B.overridecode='CAPACITY_TAG_OVRD') "])
        s2str = "".join([s2str, "order by A.customername, B.overridecode, A.accountid, B.starttime"])

        return(s2str)