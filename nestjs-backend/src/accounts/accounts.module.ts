import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { BankAccount } from './bank-account.entity';
import { BankConnection } from '../bank-connections/bank-connection.entity';
import { AccountsService } from './accounts.service';
import { AccountsController } from './accounts.controller';
import { JpmorganModule } from '../connectors/jpmorgan/jpmorgan.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([BankAccount, BankConnection]),
    JpmorganModule,
  ],
  controllers: [AccountsController],
  providers: [AccountsService],
  exports: [AccountsService],
})
export class AccountsModule {}
