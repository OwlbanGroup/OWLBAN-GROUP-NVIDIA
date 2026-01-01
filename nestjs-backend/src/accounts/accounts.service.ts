import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { BankAccount } from './bank-account.entity';
import { BankConnection } from '../bank-connections/bank-connection.entity';
import { JpmorganService } from '../connectors/jpmorgan/jpmorgan.service';

@Injectable()
export class AccountsService {
  constructor(
    @InjectRepository(BankAccount)
    private readonly accountsRepo: Repository<BankAccount>,
    @InjectRepository(BankConnection)
    private readonly connectionsRepo: Repository<BankConnection>,
    private readonly jpm: JpmorganService,
  ) {}

  async syncAccountsForConnection(connectionId: string) {
    const connection = await this.connectionsRepo.findOne({
      where: { id: connectionId },
    });
    if (!connection) {
      throw new Error('Bank connection not found');
    }

    if (connection.provider !== 'JPMORGAN') {
      throw new Error('Unsupported provider');
    }

    const jpmAccounts = await this.jpm.fetchAccounts(
      connection.providerConnectionId,
    );

    for (const jpmAcc of jpmAccounts) {
      let acc = await this.accountsRepo.findOne({
        where: {
          bankConnection: { id: connection.id },
          providerAccountId: jpmAcc.id,
        },
        relations: ['bankConnection'],
      });

      if (!acc) {
        acc = this.accountsRepo.create({
          bankConnection: connection,
          providerAccountId: jpmAcc.id,
        });
      }

      acc.name = jpmAcc.name;
      acc.type = (jpmAcc.type?.toUpperCase() as any) || 'CHECKING';
      acc.currency = jpmAcc.currency || 'USD';

      await this.accountsRepo.save(acc);
    }

    // return latest accounts
    return this.accountsRepo.find({
      where: { bankConnection: { id: connection.id } },
    });
  }

  async listAccountsByOrganization(orgId: string) {
    return this.accountsRepo
      .createQueryBuilder('acc')
      .leftJoin('acc.bankConnection', 'bc')
      .leftJoin('bc.organization', 'org')
      .where('org.id = :orgId', { orgId })
      .getMany();
  }
}
